'''
Transform user Python functions to ASTs via generating staging functions.
'''

import abc
import re
import tokenize

import freetensor_ffi as ffi

import sys
import ast
import numpy as np
import inspect
import traceback
import sourceinspect as ins
import copy
from typing import Callable, Dict, List, Sequence, Optional, Any, TypeVar, Union
from dataclasses import dataclass

from . import nodes
from .nodes import (_VarDef, VarRef, ndim, pop_ast, For, If, Else, MarkNid,
                    intrinsic, l_and, l_or, l_not, if_then_else, ctx_stack,
                    Func, Assert)

assert sys.version_info >= (3,
                            8), "Python version lower than 3.8 is not supported"


class TransformError(Exception):
    '''Error occurred during AST transforming from python function to staging function that generates IR tree.'''

    def __init__(self, message: str, filename: str, base_lineno: int,
                 error_node: ast.AST) -> None:
        super().__init__(
            f'At {filename}:{base_lineno + error_node.lineno}:\n    {message}.')


class StagingError(Exception):
    '''Error occurred during staging function execution (i.e. IR tree generation).'''

    def __init__(self, message: str) -> None:
        # TODO: add output of StagingContext.call_stack
        super().__init__(
            f'{message}:\n{"".join(traceback.format_list(StagingContext.call_stack[1:]))}'
            .lstrip())


@dataclass
class FunctionScope:
    filename: str
    funcname: str

    def __enter__(self):
        StagingContext.call_stack.append(
            traceback.FrameSummary(self.filename, 1, self.funcname))
        StagingContext.allow_return_stack.append(True)

    def __exit__(self, exc_class, exc_value, traceback):
        if exc_class is None:
            StagingContext.call_stack.pop()
        StagingContext.allow_return_stack.pop()


class NamingScope(FunctionScope):

    def __init__(self, filename: str, funcname: str,
                 namespace: Optional[str]) -> None:
        super().__init__(filename, funcname)
        if len(StagingContext.id_stack) > 0 and namespace is None:
            raise StagingError('Namespace must not be None for inner levels.')
        self.namespace = namespace
        self.ids = {}

    def __enter__(self):
        super().__enter__()
        StagingContext.id_stack.append(self)

    def __exit__(self, _1, _2, _3):
        super().__exit__(_1, _2, _3)
        popped = StagingContext.id_stack.pop()
        if popped != self:
            raise StagingError('NamingScope enter/exit not match, must be FILO')

    def fullid(self, nid: str):
        if self.namespace is not None:
            prefix = self.namespace + '->'
        else:
            prefix = ''

        if nid in self.ids:
            suffix = '$' + str(self.ids[nid])
            self.ids[nid] += 1
        else:
            suffix = ''
            self.ids[nid] = 1

        return prefix + nid + suffix


class LifetimeScope:

    def __init__(self):
        self.implicit_scopes = []

    def __enter__(self):
        StagingContext.lifetime_stack.append(self)
        StagingContext.allow_return_stack.append(False)

    def __exit__(self, _1, _2, _3):
        for scope in reversed(self.implicit_scopes):
            scope.__exit__(None, None, None)
        popped = StagingContext.lifetime_stack.pop()
        if popped != self:
            raise StagingError(
                'LifetimeScope enter/exit not match, must be FILO')
        StagingContext.allow_return_stack.pop()

    def register_implicit_scope(self, scope):
        self.implicit_scopes.append(scope)
        return scope.__enter__()


class StagingContext:
    '''Helper class managing context in IR staging.'''
    id_stack: List[NamingScope] = []
    lifetime_stack: List[LifetimeScope] = []
    allow_return_stack: List[bool] = []
    closure: Dict[str, Any] = {}
    call_stack: List[traceback.FrameSummary] = []
    name_dict: Dict[str, int] = {}

    @staticmethod
    def register_implicit_scope(scope):
        return StagingContext.lifetime_stack[-1].register_implicit_scope(scope)

    @staticmethod
    def fullid(nid: str) -> str:
        '''Get namespace-prepended full nid of given short nid.'''
        return StagingContext.id_stack[-1].fullid(nid)

    @staticmethod
    def fullname(name: str) -> str:
        '''Get distinct name.'''
        if name in StagingContext.name_dict:
            StagingContext.name_dict[name] += 1
            return f'{name}_{StagingContext.name_dict[name]}'
        else:
            StagingContext.name_dict[name] = 0
            return name

    @staticmethod
    def allow_return():
        return StagingContext.allow_return_stack[-1]

    @staticmethod
    def in_staging():
        return len(StagingContext.lifetime_stack) > 0

    @staticmethod
    def reset():
        StagingContext.id_stack.clear()
        StagingContext.lifetime_stack.clear()
        StagingContext.closure = {}
        StagingContext.call_stack = []
        StagingContext.name_dict = {}


def prepare_vardef(name: str,
                   override: bool = False,
                   capture: Optional[ffi.Array] = None):
    fullname = StagingContext.fullname(name) if not override else name
    if capture:
        StagingContext.closure[fullname] = capture
    return fullname


F = TypeVar('F')


def staged_callable(staging: F, original, doc: Optional[str] = None) -> F:

    def impl(*args, **kwargs):
        if StagingContext.in_staging():
            return staging(*args, **kwargs)
        else:
            return original(*args, **kwargs)

    # Set the name and doc of the staged function.
    # It helps when printing error messages
    impl.__name__ = staging.__name__
    if doc is not None:
        impl.__doc__ = doc

    return impl


class StagedIterable:

    def foreach(self, name: str, f: Callable[[Any], None]):
        raise NotImplementedError()


def foreach(name: str, iter, body: Callable[[Any], None]) -> None:
    '''Customized foreach wrapper.
    If `value` is instance of `StagedIterable`, its regarded as a customized foreach behavior and
    used to generate code for the python for loop.
    Otherwise, we try to execute the loop as usual.
    '''
    if isinstance(iter, StagedIterable):
        iter.foreach(name, body)
    else:
        for iter_var in iter:
            body(iter_var)


class StagedAssignable(abc.ABC):

    @abc.abstractmethod
    def assign(self, name: str) -> VarRef:
        raise NotImplementedError()


def assign_stmt(name: str, value):
    '''Customized assign wrapper.
    If `value` is instance of `StagedAssignable`, it's regarded as a customized assign behavior and
    gets executed with the assigned target variable name.
    This wrapper is used for initializing a variable.
    '''
    if isinstance(value, StagedAssignable):
        return value.assign(name)
    else:
        return value


class StagedTypeAnnotationMeta(abc.ABCMeta):

    def __getitem__(self, args):
        return self(*args)


class StagedTypeAnnotation(metaclass=StagedTypeAnnotationMeta):

    @abc.abstractmethod
    def annotate(self, name: str) -> VarRef:
        raise NotImplementedError()


def annotate_stmt(name: str, ty):
    if isinstance(ty, StagedTypeAnnotation):
        return ty.annotate(name)
    return None


def if_then_else_stmt(predicate, then_body, else_body=None):
    '''If-then-else statement staging tool.
    When predicate is deterministic in staging, only one branch is generated.
    Otherwise, a If node in IR is generated.
    '''
    if type(predicate) == bool:
        if predicate:
            then_body()
        elif else_body:
            else_body()
    else:
        with If(predicate):
            with LifetimeScope():
                then_body()
        if else_body:
            with Else():
                with LifetimeScope():
                    return else_body()


def if_then_else_expr(predicate, then_expr, else_expr):
    '''If-then-else expression staging tool.'''
    if type(predicate) == bool:
        if predicate:
            return then_expr
        else:
            return else_expr
    else:
        return if_then_else(predicate, then_expr, else_expr)


def return_stmt(value, funcname):
    '''Return staging tool. Only allow return in static control flow.'''
    if not StagingContext.allow_return():
        raise StagingError(
            'Return is only allowed in statically deterministic control flow.')
    if isinstance(value, StagedAssignable):
        value = value.assign(funcname)
    return value


def assert_stmt(test):
    '''Assert staging tool.'''
    if isinstance(test, ffi.Expr):
        StagingContext.register_implicit_scope(Assert(test))
    else:
        assert test


@dataclass
class VarCreator(StagedAssignable):
    shape: Union[Sequence, VarRef]
    dtype: str
    mtype: str

    def assign(self, name: str) -> VarRef:
        '''Customized assign behavior. Creates a VarDef with its full name.'''
        return StagingContext.register_implicit_scope(
            _VarDef(prepare_vardef(name), self.shape, self.dtype, 'cache',
                    self.mtype))


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
    return StagingContext.register_implicit_scope(
        _VarDef(prepare_vardef(name, capture=arr), arr.shape, arr.dtype,
                'input', arr.prefer_device.main_mem_type()))


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
        return StagingContext.register_implicit_scope(
            _VarDef(prepare_vardef(name), self.shape, self.dtype, self.atype,
                    self.mtype))


class dynamic_range(StagedIterable):
    '''Dynamic range that generates For loop in IR tree.'''

    def __init__(self, start, stop=None, step=1) -> None:
        '''Initialize a dynamic range. Arguments semantic identical to builtin `range`.'''
        if stop:
            self.start = start
            self.stop = stop
        else:
            self.start = 0
            self.stop = start
        self.step = step

    def foreach(self, name: str, body: Callable[[Any], None]) -> None:
        '''Customized foreach behavior. Creates a For loop.'''
        with For(StagingContext.fullname(name), self.start, self.stop,
                 self.step) as iter_var:
            with LifetimeScope():
                body(iter_var)


def boolop_expr(native_reducer, ir_reducer, lazy_args):
    result = lazy_args[0]()
    for f in lazy_args[1:]:
        if isinstance(result, ffi.Expr):
            result = ir_reducer(result, f)
        else:
            result = native_reducer(result, f)
    return result


def and_expr(*lazy_args):
    return boolop_expr(lambda a, fb: a and fb(), lambda a, fb: l_and(a, fb()),
                       lazy_args)


def or_expr(*lazy_args):
    return boolop_expr(lambda a, fb: a or fb(), lambda a, fb: l_or(a, fb()),
                       lazy_args)


def not_expr(arg):
    if not isinstance(arg, bool):
        return l_not(arg)
    else:
        return not arg


def functiondef_wrapper(filename, funcname):
    namespace = ctx_stack.top().get_next_nid()
    ctx_stack.top().set_next_nid('')
    if namespace == '':
        return FunctionScope(filename, funcname)
    else:
        return NamingScope(filename, funcname, namespace)


def metadata(entry: str):
    parts = entry.split()
    if parts[0] == 'nid:':
        ctx_stack.top().set_next_nid(StagingContext.fullid(parts[1]))
    elif parts[0] == 'no_deps:':
        var_name = parts[1]

        back = inspect.currentframe().f_back

        if var_name in back.f_locals:
            var = back.f_locals[var_name]
        elif var_name in back.f_globals:
            var = back.f_globals[var_name]
        else:
            raise StagingError(
                f'Local variable {var_name} not found for annotating comment ({entry})'
            )

        if not isinstance(var, VarRef):
            raise StagingError(
                f'Local variable {var_name} = {var} is not a VarRef, which is required by annotating comment ({entry})'
            )
        ctx_stack.top().add_next_no_deps(var.name)
    elif parts[0] == 'prefer_libs':
        ctx_stack.top().set_next_prefer_libs()


def mark_position(base_lineno: int, line_offset: int):
    original = StagingContext.call_stack[-1]
    lineno = base_lineno + line_offset - 1
    StagingContext.call_stack[-1] = traceback.FrameSummary(
        original.filename, lineno, original.name)
    if ctx_stack.top().get_next_nid() == "":
        ctx_stack.top().set_next_nid(f'{original.filename}:{lineno}')


def module_helper(callee):
    '''Helper to get an AST node with full path to given symbol, which should be in current module.'''
    return ast.Attribute(
        ast.Attribute(ast.Name('freetensor', ast.Load()), 'transformer',
                      ast.Load()), callee.__name__, ast.Load())


def call_helper(callee, *args: ast.expr, **kwargs: ast.expr):
    '''Call helper that generates a python AST Call node with given callee and arguments AST node.'''
    return ast.Call(module_helper(callee), list(args),
                    [ast.keyword(k, w) for k, w in kwargs.items()])


def function_helper(name: str, args: Sequence[str], body: List[ast.stmt],
                    nonlocals: List[str]):
    '''Function helper that generates a python AST FunctionDef node with given name, arguments name, and body.'''
    nonlocal_body = ([ast.Nonlocal(nonlocals)]
                     if len(nonlocals) > 0 else []) + body
    return ast.FunctionDef(name=name,
                           args=ast.arguments(
                               args=[],
                               vararg=None,
                               kwarg=None,
                               posonlyargs=[ast.arg(a, None) for a in args],
                               defaults=[],
                               kwonlyargs=[],
                               kw_defaults=[]),
                           body=nonlocal_body,
                           returns=None,
                           decorator_list=[])


def location_helper(new_nodes, old_node):
    if not isinstance(new_nodes, list):
        ast.copy_location(new_nodes, old_node)
        ast.fix_missing_locations(new_nodes)
    else:
        for n in new_nodes:
            ast.copy_location(n, old_node)
            ast.fix_missing_locations(n)
    return new_nodes


class NonlocalTransformingScope:

    def __init__(self, t):
        self.t: Transformer = t

    def __enter__(self):
        if self.t.nonlocals:
            self.t.nonlocals.append([])
        else:
            self.t.nonlocals = [[]]
        return [y for x in self.t.nonlocals for y in x]

    def __exit__(self, _1, _2, _3):
        self.t.nonlocals.pop()


@dataclass
class Transformer(ast.NodeTransformer):
    filename: str
    base_lineno: int
    curr_func: str = None
    nonlocals: List[List[str]] = None

    def visit(self, node: ast.AST):
        new_node = super().visit(node)
        if isinstance(node, ast.stmt) and not isinstance(node, ast.FunctionDef):
            if not isinstance(new_node, list):
                new_node = [new_node]
            return location_helper([
                ast.Expr(
                    call_helper(mark_position, ast.Constant(self.base_lineno),
                                ast.Constant(node.lineno)))
            ] + new_node, node)
        return new_node

    def visit_Assign(self, old_node: ast.Assign) -> ast.Assign:
        '''Rule:
        `lhs = rhs` -> `lhs = assign('lhs', rhs)`
        `x.lhs = rhs` -> `x.lhs = assign('lhs', rhs)`
        '''
        node: ast.Assign = self.generic_visit(old_node)
        # FIXME: multi-assign not implemented
        if len(node.targets) == 1 and (isinstance(node.targets[0], ast.Name) or
                                       isinstance(node.targets[0],
                                                  ast.Attribute)):
            name = None
            if isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
            elif isinstance(node.targets[0], ast.Attribute):
                name = node.targets[0].attr
            if name is not None:
                node = ast.Assign(
                    node.targets,
                    call_helper(assign_stmt, ast.Constant(name), node.value))
        return location_helper(node, old_node)

    def visit_AnnAssign(self, old_node: ast.AnnAssign) -> Any:
        '''Rule:
        `x: Ty` -> ```
        freetensor__annotate__x = annotate_stmt('x', Ty)
        if freetensor__annotate__x:
            x = freetensor__annotate__x
        ```: pure annotation
        '''
        node: ast.AnnAssign = self.generic_visit(old_node)
        if isinstance(node.target, ast.Name) and node.value is None:
            x = node.target
            x_str = ast.Constant(x.id)
            Ty = node.annotation

            intermediate = f'freetensor__annotate__{x.id}'
            intermediate_store = ast.Name(intermediate, ast.Store())
            intermediate_load = ast.Name(intermediate, ast.Load())
            node = [
                ast.Assign([intermediate_store],
                           call_helper(annotate_stmt, x_str, Ty)),
                ast.If(intermediate_load, [ast.Assign([x], intermediate_load)],
                       [])
            ]
        return node

    def visit_For(self, old_node: ast.For):
        '''Rule:
        ```
        for x in iter:
            body
        ```
        ->
        ```
        def for_body(x):
            body
        foreach('x', iter, for_body)
        ```'''
        if isinstance(old_node.target, ast.Name) and len(old_node.orelse) == 0:
            with NonlocalTransformingScope(self) as nonlocals:
                # while opening a fake function, For loops initiates an iter name as well.
                # need to remove it from the outer nonlocals list to implement shadowing.
                # only For loops behaves as such, so handle it specially here.
                nonlocals = set(nonlocals)
                if old_node.target.id in nonlocals:
                    nonlocals.remove(old_node.target.id)
                nonlocals = list(nonlocals)

                node = self.generic_visit(old_node)
                node = [
                    function_helper('for_body', [node.target.id], node.body,
                                    nonlocals),
                    ast.Expr(
                        call_helper(foreach,
                                    ast.Constant(node.target.id), node.iter,
                                    ast.Name('for_body', ast.Load())))
                ]
        else:
            node = self.generic_visit(old_node)
        return location_helper(node, old_node)

    def visit_Call(self, old_node: ast.Call):
        '''Rule:
        `range(...)` -> `dynamic_range(...)`
        '''
        node: ast.Call = self.generic_visit(old_node)
        if isinstance(node.func, ast.Name):
            if node.func.id == 'range':
                node = ast.Call(module_helper(dynamic_range), node.args,
                                node.keywords)
        return location_helper(node, old_node)

    def visit_If(self, old_node: ast.If):
        '''Rule:
        ```
        if pred:
            body
        else:
            orelse
        ```
        ->
        ```
        def then_body():
            body
        def else_body():
            orelse
        if_then_else_stmt(pred, then_body, else_body)
        ```
        '''
        test = self.visit(old_node.test)
        with NonlocalTransformingScope(self) as nonlocals:
            new_node = [
                function_helper('then_body', [], [
                    z for x in old_node.body for y in [self.visit(x)]
                    for z in (y if isinstance(y, list) else [y])
                ], nonlocals)
            ]
        then_body = ast.Name('then_body', ast.Load())
        if old_node.orelse:
            with NonlocalTransformingScope(self) as nonlocals:
                new_node.append(
                    function_helper('else_body', [], [
                        z for x in old_node.orelse for y in [self.visit(x)]
                        for z in (y if isinstance(y, list) else [y])
                    ], nonlocals))
            else_body = ast.Name('else_body', ast.Load())
        else:
            else_body = ast.Constant(None)
        new_node.append(
            ast.Expr(call_helper(if_then_else_stmt, test, then_body,
                                 else_body)))
        return location_helper(new_node, old_node)

    def visit_IfExp(self, old_node: ast.IfExp):
        '''Rule: `body if test else orelse` -> `if_then_else_expr(test, body, orelse)`'''
        node = self.generic_visit(old_node)
        node = call_helper(if_then_else_expr, node.test, node.body, node.orelse)
        return location_helper(node, old_node)

    def visit_FunctionDef(self, old_node: ast.FunctionDef) -> Any:
        prev_func = self.curr_func
        self.curr_func = old_node.name

        # nested functions follow original Python (shitty) scoping,
        # thus backup the nonlocals stack and prepare a clean one.
        prev_nonlocals = self.nonlocals
        self.nonlocals = None

        with NonlocalTransformingScope(self):
            # mark arguments as nonlocal
            for name in old_node.args.args + old_node.args.kwonlyargs:
                self.nonlocals[-1].append(name.arg)
            if old_node.args.vararg:
                self.nonlocals[-1].append(old_node.args.vararg.arg)
            if old_node.args.kwarg:
                self.nonlocals[-1].append(old_node.args.kwarg.arg)

            node: ast.FunctionDef = self.generic_visit(old_node)
            node.decorator_list = []
            node.body = [
                ast.With(
                    items=[
                        ast.withitem(context_expr=call_helper(
                            functiondef_wrapper, ast.Constant(self.filename),
                            ast.Constant(node.name)),
                                     optional_vars=None)
                    ],
                    body=[
                        stmt for arg in node.args.posonlyargs + node.args.args
                        if arg.annotation for stmt in self.visit_AnnAssign(
                            location_helper(
                                ast.AnnAssign(ast.Name(arg.arg, ast.Store(
                                )), arg.annotation, None, 1), old_node))
                    ] + node.body)
            ]

        self.curr_func = prev_func
        self.nonlocals = prev_nonlocals
        return location_helper(node, old_node)

    def visit_Assert(self, old_node: ast.Assert) -> Any:
        node: ast.Assert = self.generic_visit(old_node)
        node = ast.Expr(call_helper(assert_stmt, node.test))
        return location_helper(node, old_node)

    def visit_BoolOp(self, old_node: ast.BoolOp) -> Any:
        node: ast.BoolOp = self.generic_visit(old_node)
        if isinstance(node.op, ast.And):
            libfunc = and_expr
        elif isinstance(node.op, ast.Or):
            libfunc = or_expr
        else:
            return location_helper(node, old_node)
        empty_args = ast.arguments(args=[],
                                   vararg=None,
                                   kwarg=None,
                                   posonlyargs=[],
                                   defaults=[],
                                   kwonlyargs=[],
                                   kw_defaults=[])
        node = call_helper(libfunc,
                           *[ast.Lambda(empty_args, v) for v in node.values])
        return location_helper(node, old_node)

    def visit_UnaryOp(self, old_node: ast.UnaryOp) -> Any:
        node: ast.UnaryOp = self.generic_visit(old_node)
        if isinstance(node.op, ast.Not):
            node = call_helper(not_expr, node.operand)
        return location_helper(node, old_node)

    def visit_Compare(self, old_node: ast.Compare) -> Any:
        '''Expand multiple comparison into `and` expression.'''
        if len(old_node.comparators) == 1:
            return self.generic_visit(old_node)
        lhs = old_node.left
        node = ast.BoolOp(ast.And(), [])
        for op, rhs in zip(old_node.ops, old_node.comparators):
            node.values.append(ast.Compare(lhs, [op], [rhs]))
            lhs = rhs
        return self.visit(location_helper(node, old_node))

    def visit_Return(self, old_node: ast.Return) -> Any:
        node: ast.Return = self.generic_visit(old_node)
        assert self.curr_func is not None
        node = ast.Return(
            call_helper(return_stmt, node.value, ast.Constant(self.curr_func)))
        return location_helper(node, old_node)

    def visit_Lambda(self, old_node: ast.Lambda) -> Any:
        with NonlocalTransformingScope(self):
            node: ast.Lambda = self.generic_visit(old_node)
        return location_helper(node, old_node)

    def visit_comprehension(self, old_node: ast.comprehension) -> Any:
        with NonlocalTransformingScope(self):
            node: ast.comprehension = self.generic_visit(old_node)
        return location_helper(node, old_node)

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Store):
            self.nonlocals[-1].append(node.id)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        raise TransformError('Async functions not supported.', self.filename,
                             self.base_lineno, node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        raise TransformError('Class definitions not supported.', self.filename,
                             self.base_lineno, node)


def _remove_indent(lines: List[str]) -> str:
    spaces_to_remove = next((i for i, x in enumerate(lines[0]) if x != ' '),
                            len(lines[0]))
    return ''.join(line[spaces_to_remove:] for line in lines)


def _get_caller_env(depth: int):
    frame = inspect.currentframe()
    try:
        parent = frame
        for _ in range(depth + 1):
            parent = parent.f_back
        caller_env = copy.copy(parent.f_globals)
        caller_env.update(parent.f_locals)
    finally:
        del frame
    return caller_env


def process_annotating_comments(src: str):
    new_src = []
    for line in src.splitlines():
        indent = re.match('\\s*', line)[0]
        rest_line = line[len(indent):]
        if rest_line.startswith('#! '):
            arg = rest_line[3:].replace('"', '\\"')
            new_src.append(f'{indent}freetensor.metadata("{arg}")')
        else:
            new_src.append(line)
    new_src = '\n'.join(new_src)
    return new_src


def into_staging(func, caller_env, src: str = None, verbose=False):
    if src is None:
        lines, lineno = ins.getsourcelines(func)
        src = _remove_indent(lines)
        file = ins.getfile(func)
    else:
        lineno = 1
        file = f'<staging:{func.__name__}>'

    tree = ast.parse(process_annotating_comments(src))
    tree = ast.fix_missing_locations(Transformer(file, lineno).visit(tree))

    import astor
    source = astor.to_source(tree)

    if verbose:
        from pygments import highlight
        from pygments.lexers import PythonLexer
        from pygments.formatters import TerminalFormatter
        print(highlight(source, PythonLexer(),
                        TerminalFormatter(bg='dark', linenos=True)),
              file=sys.stderr)

    caller_env['freetensor'] = sys.modules['freetensor']
    exec(compile(source, f'<staging:{func.__name__}>', 'exec'), caller_env)
    return caller_env[func.__name__], file, func.__name__


def transform(func=None, verbose: int = 0, depth: int = 1, caller_env=None):
    '''
    Transform a user function to an AST

    Parameters
    ----------
    func : Python function
        The user function to transform. If not specified, a partial function will
        be returend, which can be used as a decorator
    verbose : int
        0 = print nothing. 1 = print the resulting AST. 2 = 1 + print the generated
        Python code that is used for transforming
    '''

    if verbose is None:
        verbose = 0
    if caller_env is None:
        caller_env = _get_caller_env(depth)

    def decorator(func):
        params = list(inspect.signature(func).parameters)
        staging_func, filename, funcname = into_staging(func,
                                                        caller_env,
                                                        verbose=verbose >= 2)

        try:
            with LifetimeScope():
                with NamingScope(filename, funcname, None):
                    # for p in params:
                    #     StagingContext.id_stack[-1].ids[p] = 1
                    #     StagingContext.name_dict[p] = 0
                    returns = staging_func(*params)
                    if isinstance(returns, VarRef):
                        returns = [returns]
                    elif isinstance(returns, tuple):
                        for ret in returns:
                            if not isinstance(ret, VarRef):
                                raise StagingError(
                                    'Illegal return at top level, need to be a `VarRef` or a tuple of `VarRef`s'
                                )
                        returns = list(returns)
                    elif returns is None:
                        returns = []
                    else:
                        raise StagingError(
                            'Illegal return at top level, need to be a `VarRef` or a tuple of `VarRef`s'
                        )
                    for ret in returns:
                        if ret.vardef.atype != ffi.AccessType('inout'):
                            ret.vardef.set_atype('output')
                    returns = [
                        (ret.vardef.name, ret.vardef.dtype) for ret in returns
                    ]

                    closure = StagingContext.closure
        except Exception as e:
            raise StagingError('Exception occurred in staging') from e
        finally:
            StagingContext.reset()
            staged_ast = pop_ast()

        staged = Func(func.__name__, params + list(closure.keys()), returns,
                      staged_ast, closure)

        if verbose >= 1:
            print("The transformed AST is:", file=sys.stderr)
            print(staged, file=sys.stderr)
            print(file=sys.stderr)

        return staged

    if callable(func):
        return decorator(func)
    else:
        return decorator


def inline(func=None, src=None, fallback=None, verbose=False, caller_env=None):
    '''
    Enable a user function to be called by a transformed function at run time

    Parameters
    ----------
    func : Python function
        The user function
    src : str (Optional)
        The source code of `func`. This parameter is only required if the source
        code cannot be get automatically, e.g., if `func` is generated from a `exec`
    verbose : bool
        True to print the generated Python code that is used for transforming
    '''

    if caller_env is None:
        caller_env = _get_caller_env(1)

    def decorator(func):
        return staged_callable(
            into_staging(func, caller_env, src, verbose=verbose)[0], fallback or
            func)

    if callable(func):
        return decorator(func)
    else:
        return decorator
