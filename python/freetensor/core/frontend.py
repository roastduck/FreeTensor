'''
A frontend transforming user Python functions to ASTs via staging.
'''

import sys
import numpy as np
import inspect
import traceback
from numbers import Number
from typing import Callable, Dict, List, Sequence, Optional, Any, Union
from dataclasses import dataclass

import freetensor_ffi as ffi

from .expr import (dtype, mtype, ndim, l_and, l_or, l_not, if_then_else, shape,
                   VarVersionRef)
from .stmt import (_VarDef, VarRef, For, If, Else, ctx_stack, Assert, Invoke,
                   MarkVersion, UserGradStaged)
from .staging import (StagedPredicate, StagedTypeAnnotation, StagedAssignable,
                      StagedIterable, StagingOverload, StagingOverloadStack,
                      StagingError)
from .context import StmtRange

assert sys.version_info >= (3, 8), \
    "Python version lower than 3.8 is not supported"


def staged_callable(staging, original, doc: Optional[str] = None):

    def impl(*args, **kwargs):
        if lang_overload.in_staging():
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
        if lang_overload.in_staging():
            lang_overload.lifetime_stack.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for scope in reversed(self.inner_scopes):
            scope.__exit__(exc_type, exc_val, exc_tb)
        if lang_overload.in_staging():
            popped = lang_overload.lifetime_stack.pop()
            if popped != self:
                raise lang_overload.error(
                    'LifetimeScope enter/exit not match, must be FILO')

    def register_inner_scope(self, scope):
        self.inner_scopes.append(scope)
        return scope.__enter__()


class FreeTensorStagingError(StagingError):
    '''Error occurred during staging function execution (i.e. IR tree generation).'''

    def __init__(self, message: str) -> None:
        # TODO: add output of StagingContext.call_stack
        super().__init__(
            f'{message}:\n{"".join(traceback.format_list(ctx_stack.debug_call_stack))}'
            .lstrip())


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
                        capture=None):
        fullname = self.fullname(name)
        if capture:
            self.closure[fullname] = capture

        return self.lifetime_stack[-1].register_inner_scope(
            _VarDef(fullname, shape, dtype, atype, mtype))

    def register_inlined_invoke(self, ret_names: Sequence[str], func: ffi.Func,
                                args, kvs):
        ret_names = [self.fullname(name) for name in ret_names]
        return self.lifetime_stack[-1].register_inner_scope(
            Invoke(ret_names, func, args, kvs))

    def register_assert(self, pred):
        self.lifetime_stack[-1].register_inner_scope(Assert(pred))

    def fullname(self, name: str) -> str:
        '''Get distinct name.'''
        if name in self.name_dict:
            self.name_dict[name] += 1
            return self.fullname(f'{name}_{self.name_dict[name]}')
        else:
            self.name_dict[name] = 0
            return name

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

            # Push debug call stack with some random line number.
            # It will be updated by `mark_position` calls in the function.
            ctx_stack.debug_call_stack.append(
                traceback.FrameSummary(filename, 1, func.__name__))

            result = basic_wrapped(*args, **kwargs)

            # Pop debug call stack.
            ctx_stack.debug_call_stack.pop()

            ctx_stack.top().set_caller_metadata(prev)

            return result

        return wrapped

    def metadata(self, entry: str) -> None:
        parts = entry.split()
        if len(parts) == 0:
            return

        key = parts[0]
        val = None
        if len(parts) > 1:
            val = parts[1]

        if key == 'label:':
            if val is not None:
                ctx_stack.top().add_label(val)
                return
        elif key == 'no_deps:':
            if val is not None:
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
                return
        elif key == 'prefer_libs':
            ctx_stack.top().set_next_prefer_libs()
            return

        raise ffi.InvalidProgram('''Invalid metadata. Possible metadata are:
`label: <label_name>`: to label the following statement,
`no_deps: <variable_name>`: to mark a variable to have no dependence along the following loop,
`prefer_libs`: to indicate the following statement should preferably be executed using external libraries.
''')

    def mark_position(self, lineno: int):
        ctx_stack.mark_position(lineno)

    def error(self, content: str):
        return FreeTensorStagingError(content)


class FreeTensorOverloadStack(StagingOverloadStack):

    def __init__(self):
        super().__init__(FreeTensorOverload)


lang_overload: FreeTensorOverloadStack = FreeTensorOverloadStack()


def _register_as_predicate(ty):

    def _logical_and(a: ty, fb: Callable[[], StagedPredicate]):
        return l_and(a, fb())

    def _logical_or(a: ty, fb: Callable[[], StagedPredicate]):
        return l_or(a, fb())

    def _logical_not(a: ty):
        return l_not(a)

    def _if_then_else_stmt(pred: ty, then_body: Callable[[], None],
                           else_body: Optional[Callable[[], None]]):
        with lang_overload.allow_shortcut_scope(False):
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
        lang_overload.register_assert(pred)

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
    assigned: bool = False

    def assign(self, name: str) -> VarRef:
        '''Customized assign behavior. Creates a VarDef with its full name.'''
        if not self.assigned:
            self.assigned = True
            return lang_overload.register_vardef(name, self.shape, self.dtype,
                                                 'cache', self.mtype)
        else:
            raise lang_overload.error(
                "Create new tensors in an `a = b = c`-like multi-assignment "
                "is not supported")


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
    return lang_overload.register_vardef(name,
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
        return lang_overload.register_vardef(name, self.shape, self.dtype,
                                             self.atype, self.mtype)


@dataclass
class VersionMarker(StagedAssignable):
    var: VarRef
    assigned: bool = False

    def assign(self, tape_name: str) -> VarRef:
        '''Customized assign behavior. Creates a MarkVersion with its full name.'''
        if not self.assigned:
            self.assigned = True
            full_tape_name = lang_overload.fullname(tape_name)
            MarkVersion(full_tape_name, self.var)
            return VarVersionRef(full_tape_name, self.var.full_shape,
                                 self.var.dtype, self.var.mtype,
                                 self.var.indices)
        else:
            raise lang_overload.error(
                "Marking version in an `a = b = c`-like multi-assignment is not"
                " supported")


def push_for_backward(var: VarRef):
    '''
    Push the current value from the forward pass to be used at the backward pass

    This function is for custom gradients. See `UserGrad` for details on how to provide custom
    gradients.

    You may imagine there is a virtual stack for each variable. Each time you call `x_handle =
    push_for_backward(x)` in the forward pass, the value of `x` **at the current iteration**
    will be "pushed" to the virtual stack. You can access `x_handle` at the backward pass. Each
    time you access `x_handle`, you will "pop" the stack and get the value of `x` **pushed at
    the same iteration**. Since the "stack" is virtual, you do NOT need to "pop" the same count
    as "push"es: the version numbering is fully automatic. Besides, there may not be a real
    stack at runtime: it can be compiled to any data structure.

    This function will be staged to `mark_version` statement in the IR.
    '''

    return VersionMarker(var)


class UserGrad(UserGradStaged):
    '''
    Define a custom gradient

    Follow the following steps to define custom gradient:

    1. Add some `mark_version` statements in the program. `mark_version('y0', y)` marks the specific
    versions of variable `y` **at the program position of the statement** and **at all iterations**
    as `'y0'`.
    2. Add a `UserGrad` scope.
    2.1. `UserGrad` optionally receives parameter `stmt_range`, recorded by the `StmtRange` helper class,
    which means the gradient is for the code specified in the range. Ignoring the parameter means setting
    gradient for the previous statement of the scope.
    2.2. Other parameters of `UserGrad` sets the mapping from original variables to gradient variables.
    `with UserGradForPrevStmt(x, y) as (dx, dy)` provides `VarRef` `dx` and `dy` as gradient variables
    to be used inside the scope.
    3. In order to use the value from the forward pass in the backward pass, do not access the forward
    variables directly in the scope. Instead, use `load_at_version` expressions. `load_at_version(y0, i, j)`
    loads from `y[i, j]` **at the specific version marked by `y0 = mark_version(y)`**, saved from **the same
    iteration in the forward pass**. (If directly writing staged code, it is `MarkVersion('y0', y)`). In
    other words, after AD, the position of `mark_version` and the dynamic loop iterator together makes up
    the actual version number for the tape.
    4. Build the AST with `pop_ast_and_user_grads` instead of `pop_ast`. An extra list will be returned
    together with the AST, which you need to pass as `grad`'s `user_grads` argument. This list records
    the forward-to-backward relation of the nodes.

    If you are directly writing staged code, use `UserGradStaged` instead.

    Parameters
    ----------
    *args: Sequence[VarRef]
        (Positional variadic) Mapping from original variables to gradient variables
    stmt_range: Optional[StmtRange]
        The range in the original program that we are setting custom gradient for
    '''

    def __init__(self, *args: Sequence[VarRef], stmt_range: StmtRange = None):
        super(UserGrad, self).__init__(*args, stmt_range=stmt_range)
        self.lifetime_scope = LifetimeScope()

    def __enter__(self):
        ret = super(UserGrad, self).__enter__()
        self.lifetime_scope.__enter__()
        return ret

    def __exit__(self, exc_type, exc_value, traceback):
        self.lifetime_scope.__exit__(exc_type, exc_value, traceback)
        return super(UserGrad, self).__exit__(exc_type, exc_value, traceback)


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
            raise lang_overload.error(
                'dynamic_range only supports exactly one target variable')

        # Early optimizations
        if isinstance(self.start, Number) and isinstance(
                self.stop, Number) and isinstance(self.step, Number):
            if not range(self.start, self.stop, self.step):
                return
            if len(range(self.start, self.stop, self.step)) == 1:
                with LifetimeScope():
                    body(self.start)
                return

        with lang_overload.allow_shortcut_scope(False):
            with For(lang_overload.fullname(name), self.start, self.stop,
                     self.step) as iter_var:
                with LifetimeScope():
                    body(iter_var)


static_range = range
