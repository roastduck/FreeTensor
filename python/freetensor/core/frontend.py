'''
A frontend transforming user Python functions to ASTs via staging.
'''

import sys
import numpy as np
import functools
import inspect
from typing import Callable, Dict, List, Sequence, Optional, Any, Union
from dataclasses import dataclass

import freetensor_ffi as ffi

from .context import pop_ast
from .expr import (dtype, mtype, ndim, intrinsic, l_and, l_or, l_not,
                   if_then_else, shape)
from .stmt import (_VarDef, NamedScope, VarRef, For, If, Else, MarkLabel,
                   ctx_stack, Func, Assert)

from .staging import (StagedPredicate, StagedTypeAnnotation, StagedAssignable,
                      StagedIterable, StagingError, StagingOverload,
                      TransformError)

assert sys.version_info >= (3, 8), \
    "Python version lower than 3.8 is not supported"


def staged_callable(staging, original, doc: Optional[str] = None):

    def impl(*args, **kwargs):
        if _overload.in_staging():
            return staging(*args, **kwargs)
        else:
            return original(*args, **kwargs)

    # Set the name and doc of the staged function.
    # It helps when printing error messages
    impl.__name__ = staging.__name__
    if doc is not None:
        impl.__doc__ = doc

    return impl


class LifetimeScope:
    '''This scope is used to register multiple scopes inside a single lifetime scope.
    The inner scopes might be used to register variables, etc.
    They will be exited in reverse order of their registration.
    '''

    def __init__(self):
        self.inner_scopes = []

    def __enter__(self):
        _overload.lifetime_stack.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for scope in reversed(self.inner_scopes):
            scope.__exit__(exc_type, exc_val, exc_tb)
        popped = _overload.lifetime_stack.pop()
        if popped != self:
            raise _overload.error(
                'LifetimeScope enter/exit not match, must be FILO')

    def register_inner_scope(self, scope):
        self.inner_scopes.append(scope)
        return scope.__enter__()


class FreeTensorOverload(StagingOverload):
    '''Helper class managing context in IR staging.'''

    def __init__(self):
        super().__init__()
        self.lifetime_stack: List[LifetimeScope] = []
        self.closure: Dict[str, Any] = {}
        self.name_dict: Dict[str, int] = {}

    def register_vardef(self,
                        name,
                        shape,
                        dtype,
                        atype,
                        mtype=None,
                        capture=None,
                        override=False):
        fullname = self.fullname(name) if not override else name
        if capture:
            self.closure[fullname] = capture

        return self.lifetime_stack[-1].register_inner_scope(
            _VarDef(fullname, shape, dtype, atype, mtype))

    def register_assert(self, pred):
        self.lifetime_stack[-1].register_inner_scope(Assert(pred))

    def fullname(self, name: str) -> str:
        '''Get distinct name.'''
        if name in self.name_dict:
            self.name_dict[name] += 1
            return f'{name}_{self.name_dict[name]}'
        else:
            self.name_dict[name] = 0
            return name

    def in_staging(self,):
        return len(self.lifetime_stack) > 0

    def custom_attr(self, obj: Any, attr: str) -> Any:
        if attr == "ndim":
            return ndim(obj)
        if attr == "shape":
            return lambda i=None: shape(obj, i)
        if attr == "dtype":
            return dtype(obj)
        if attr == "mtype":
            return mtype(obj)
        raise AttributeError()

    def functiondef_wrapper(self, filename: str, func):
        basic_wrapped = super().functiondef_wrapper(filename, func)

        def wrapped(*args, __freetensor_transform_outermost__=False, **kwargs):
            if __freetensor_transform_outermost__:
                call_metadata = None
            else:
                call_metadata = ctx_stack.top().get_metadata()
            ctx_stack.top().clear_metadata()

            prev = ctx_stack.top().caller_metadata
            ctx_stack.top().set_caller_metadata(call_metadata)
            result = basic_wrapped(*args, **kwargs)
            ctx_stack.top().set_caller_metadata(prev)

            return result

        return wrapped

    def metadata(self, entry: str) -> None:
        parts = entry.split()
        if len(parts) == 0:
            return

        key = parts[0]
        if len(parts) > 1:
            key = key[:-1]
            val = parts[1]

        if key == 'label':
            ctx_stack.top().add_label(val)
        elif key == 'no_deps':
            back = inspect.currentframe().f_back

            if val in back.f_locals:
                var = back.f_locals[val]
            elif val in back.f_globals:
                var = back.f_globals[val]
            else:
                raise self.error(
                    f'Variable {val} not found for annotating comment ({key}: {val})'
                )

            if not isinstance(var, VarRef):
                raise self.error(
                    f'Variable {val} = {var} is not a VarRef, which is required by annotating comment ({key}: {val})'
                )
            ctx_stack.top().add_next_no_deps(var.name)
        elif key == 'prefer_libs':
            ctx_stack.top().set_next_prefer_libs()

    def at_position(self, filename: str, lineno: int) -> None:
        ctx_stack.top().set_next_location(filename, lineno)


_overload: FreeTensorOverload = FreeTensorOverload()


def _register_as_predicate(ty):

    def _logical_and(a: ty, fb: Callable[[], StagedPredicate]):
        return l_and(a, fb())

    def _logical_or(a: ty, fb: Callable[[], StagedPredicate]):
        return l_or(a, fb())

    def _logical_not(a: ty):
        return l_not(a)

    def _if_then_else_stmt(pred: ty, then_body: Callable[[], None],
                           else_body: Optional[Callable[[], None]]):
        with _overload.allow_shortcut_scope(False):
            with If(pred):
                with LifetimeScope():
                    then_body()
            if else_body:
                with Else():
                    with LifetimeScope():
                        else_body()

    def _if_then_else_expr(pred: ty, then_expr: Callable[[], VarRef],
                           else_expr: Callable[[], VarRef]):
        return if_then_else(pred, then_expr(), else_expr())

    def _while_stmt(pred: ty, body: Callable[[], None]):
        raise NotImplementedError()

    def _assert_stmt(pred: ty):
        _overload.register_assert(pred)

    StagedPredicate.register(ty)
    ty.logical_and = _logical_and
    ty.logical_or = _logical_or
    ty.logical_not = _logical_not
    ty.assert_stmt = _assert_stmt
    ty.if_then_else_stmt = _if_then_else_stmt
    ty.if_then_else_expr = _if_then_else_expr
    ty.while_stmt = _while_stmt


_register_as_predicate(VarRef)
_register_as_predicate(ffi.Expr)


@dataclass
class VarCreator(StagedAssignable):
    shape: Union[Sequence, VarRef]
    dtype: str
    mtype: str

    def assign(self, name: str) -> VarRef:
        '''Customized assign behavior. Creates a VarDef with its full name.'''
        return _overload.register_vardef(name, self.shape, self.dtype, 'cache',
                                         self.mtype)


def empty_staging(shape, dtype, mtype=None):
    return VarCreator(shape, dtype, mtype)


def empty_fallback(shape, dtype, mtype=None):
    return np.zeros(shape, dtype)


empty = staged_callable(
    empty_staging, empty_fallback, '''
Create an empty variable

Parameters
----------
shape : Sequence[Expr] or Var
    Shape of the variable. A variable can be created using a literal shape,
    or another fixed-length VarRef as a shape
dtype : str or DataType
    Data type of the variable
mtype : str or MemType (Optional)
    Memory type of the variable. If omitted, the main memory type of the
    default Target in config will be used
''')


class PredefinedVarCreator(VarCreator):

    def __init__(self, initializer: List[Any], dtype: str, mtype: str):

        def get_shape(lst):
            if not isinstance(lst, list):
                assert ndim(lst) == 0
                return ()

            if len(lst) == 0:
                return (0,)

            shape_ = get_shape(lst[0])
            for x in lst[1:]:
                assert shape_ == get_shape(x)

            return (len(lst),) + shape_

        super().__init__(get_shape(initializer), dtype, mtype)
        self.initializer = initializer

    def assign(self, name: str) -> VarRef:
        var = super().assign(name)

        def impl(var_slice, init_slice):
            if not isinstance(init_slice, list):
                var_slice[()] = init_slice
            else:
                for i, x in enumerate(init_slice):
                    impl(var_slice[i], x)

        impl(var, self.initializer)
        return var


def var_staging(initializer, dtype, mtype=None):
    return PredefinedVarCreator(initializer, dtype, mtype)


def var_fallback(initializer, dtype, mtype=None):
    return np.array(initializer, dtype=dtype)


var = staged_callable(
    var_staging, var_fallback, '''
Create an with variable a given initializer

Parameters
----------
initializer : Sequence[Sequence[...Sequence[Expr]...]]
    (Multi-level of) sequence of expressions. Will be data of the variable
shape : Sequence[Expr] or Var
    Shape of the variable. A variable can be created using a literal shape,
    or another fixed-length VarRef as a shape
dtype : str or DataType
    Data type of the variable
mtype : str or MemType (Optional)
    Memory type of the variable. If omitted, the main memory type of the
    default Target in config will be used
''')


def capture_var_fallback(arr: ffi.Array, name: str = 'captured'):
    return arr.numpy()


def capture_var_staging(arr: ffi.Array, name: str = 'captured'):
    return _overload.register_vardef(name,
                                     arr.shape,
                                     arr.dtype,
                                     'input',
                                     capture=arr)


capture_var = staged_callable(capture_var_staging, capture_var_fallback,
                              '''Capture external array as tensor variable.''')


class Var(StagedTypeAnnotation):

    def __init__(self, shape, dtype, atype="input", mtype=None):
        '''
        Declare a variable

        Parameters
        ----------
        name : str
            Name of the variable
        shape : Sequence[Expr] or Var
            Shape of the variable. A variable can be created using a literal shape,
            or another fixed-length VarRef as a shape
        dtype : str or DataType
            Data type of the variable
        atype : str or AccessType
            Access type of the variable. It specifies whether (and how) the variable
            is an I/O variable of the function it belongs to. Defaults to "input"
        mtype : str or MemType (Optional)
            Memory type of the variable. If omitted, the main memory type of the
            default Target in config will be used
        '''
        self.shape, self.dtype, self.atype, self.mtype = shape, dtype, atype, mtype

    def annotate(self, name: str) -> VarRef:
        return _overload.register_vardef(name, self.shape, self.dtype,
                                         self.atype, self.mtype)


class dynamic_range(StagedIterable):
    '''Dynamic range that generates For loop in IR tree.'''

    def __init__(self, start, stop=None, step=1) -> None:
        '''Initialize a dynamic range.
        Arguments semantic identical to builtin `range`.'''
        if stop:
            self.start = start
            self.stop = stop
        else:
            self.start = 0
            self.stop = start
        self.step = step

    def foreach(self, name, body: Callable[[Any], None]) -> None:
        '''Customized foreach behavior. Creates a For loop.'''
        if not isinstance(name, str):
            raise _overload.error(
                'dynamic_range only supports exactly one target variable')
        with _overload.allow_shortcut_scope(False):
            with For(_overload.fullname(name), self.start, self.stop,
                     self.step) as iter_var:
                with LifetimeScope():
                    body(iter_var)


static_range = range


def _prepare_extra_locals(default_dynamic_range):
    extra_locals = {'__ft__': sys.modules['freetensor']}
    if default_dynamic_range:
        extra_locals['range'] = dynamic_range
    return extra_locals


def transform(func=None, default_dynamic_range=True, verbose: int = 0):
    '''
    Transform a user function to an AST

    Parameters
    ----------
    func : Python function
        The user function to transform. If not specified, a partial function will
        be returend, which can be used as a decorator
    default_dynamic_range : bool
        If True, the built-in range is replaced with freetensor.dynamic_range.
        Defaults to True
    verbose : int
        0 = print nothing. 1 = print the resulting AST. 2 = 1 + print the generated
        Python code that is used for transforming
    '''

    if func is None:
        return functools.partial(transform,
                                 default_dynamic_range=default_dynamic_range,
                                 verbose=verbose)

    if verbose is None:
        verbose = 0

    extra_locals = _prepare_extra_locals(default_dynamic_range)

    params = list(inspect.signature(func).parameters)

    staging_func = _overload.into_staging(func,
                                          extra_locals,
                                          verbose=verbose >= 2)

    try:
        # Initialize _overload to prepare for staging.
        _overload.__init__()
        # Create a new scope for the function
        with LifetimeScope():
            # Run staging function with the tensor program arguments' names as parameters
            returns = staging_func(*params,
                                   __freetensor_transform_outermost__=True)
            # Check returned vardefs (if any)
            if isinstance(returns, VarRef):
                returns = [returns]
            elif isinstance(returns, tuple):
                for ret in returns:
                    if not isinstance(ret, VarRef):
                        raise _overload.error(
                            'Illegal return at top level, need to be a `VarRef` or a tuple of `VarRef`s'
                        )
                returns = list(returns)
            elif returns is None:
                returns = []
            else:
                raise _overload.error(
                    'Illegal return at top level, need to be a `VarRef` or a tuple of `VarRef`s'
                )
            # Set returned vardefs' access type to inout/output according to whether it was an input
            for ret in returns:
                if ret.vardef.atype == 'input' or ret.vardef.atype == 'inout':
                    ret.vardef.set_atype('inout')
                else:
                    ret.vardef.set_atype('output')
            returns = [(ret.vardef.name, ret.vardef.dtype) for ret in returns]

            # Set closure; they are from captured Arrays.
            closure = _overload.closure
    except StagingError:
        raise
    except TransformError:
        raise
    except Exception as e:
        raise _overload.error('Exception occurred in staging') from e
    finally:
        # Despite whether the exception is raised, we need to clean up the ctx_stack
        staged_ast = pop_ast()

    staged = Func(func.__name__, params + list(closure.keys()), returns,
                  staged_ast, closure)

    if verbose >= 1:
        print("The transformed AST is:", file=sys.stderr)
        print(staged, file=sys.stderr)
        print(file=sys.stderr)

    return staged


def inline(func=None,
           src=None,
           fallback=None,
           default_dynamic_range=True,
           verbose=False):
    '''
    Enable a user function to be called by a transformed function at run time

    Parameters
    ----------
    func : Python function
        The user function
    src : str (Optional)
        The source code of `func`. This parameter is only required if the source
        code cannot be get automatically, e.g., if `func` is generated from a `exec`
    default_dynamic_range : bool
        If True, the built-in range is replaced with freetensor.dynamic_range.
        Defaults to True
    verbose : bool
        True to print the generated Python code that is used for transforming
    '''

    if func is None:
        return functools.partial(inline,
                                 src=src,
                                 fallback=fallback,
                                 default_dynamic_range=default_dynamic_range,
                                 verbose=verbose)

    extra_locals = _prepare_extra_locals(default_dynamic_range)

    # Do not initialize _overload here, since `into_staging` does not use the context.
    # Keep the context as-is to support adding new inline functions during transforming.
    # Such a case occurs when a transformed function dynamically imports a new inline.
    transformed = _overload.into_staging(func,
                                         extra_locals,
                                         src,
                                         verbose=verbose)

    return functools.wraps(func)(staged_callable(transformed, fallback or func))
