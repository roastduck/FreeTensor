'''
New transformer implementation based on generating a staging function.
'''

import collections

import ffi

import sys
import ast
import numpy as np
import inspect
import sourceinspect as ins
import copy
from typing import Callable, Dict, List, Sequence, Optional, Mapping, Any, Tuple, Union
from dataclasses import dataclass

from . import nodes
from .nodes import (_VarDef, Var, pop_ast, For, If, Else, MarkNid, intrinsic,
                    l_and, l_or, l_not, if_then_else, ctx_stack,
                    Func, Assert)
from .utils import *

assert sys.version_info >= (3,
                            8), "Python version lower than 3.8 is not supported"


class TransformError(Exception):
    '''Error occurred during AST transforming from python function to staging function that generates IR tree.'''

    def __init__(self, message: str, error_node: ast.AST) -> None:
        # TODO: better error report
        super().__init__(
            f'At line {error_node.lineno}, column {error_node.col_offset}: {message}')


class StagingError(Exception):
    '''Error occurred during staging function execution (i.e. IR tree generation).'''

    def __init__(self, message: str) -> None:
        # TODO: better error report
        super().__init__(message)


class StagingScope:
    '''Helper class managing scopes in staging an IR tree.'''

    def __init__(self, namespace: str, allow_return: bool, borrow: bool):
        previous = StagingContext.top()

        if borrow:
            self.names = previous.names
        else:
            self.names = {}

        prev_ns = previous.namespace if previous else None
        if namespace:
            if prev_ns is None:
                self.namespace = namespace
            else:
                self.namespace = prev_ns + ':' + namespace
        else:
            self.namespace = prev_ns

        self.allow_return = allow_return

        if borrow:
            self.implicit_scopes = previous.implicit_scopes
        else:
            self.implicit_scopes = []

        self.borrow = borrow

    def __enter__(self):
        StagingContext.push(self)

    def __exit__(self, _1, _2, _3):
        if not self.borrow:
            for scope in reversed(self.implicit_scopes):
                scope.__exit__(None, None, None)
        popped = StagingContext.pop()
        if popped is not self:
            raise StagingError(
                'StagingScope enter/exit not match, must be FILO')

    def fullname(self, name: str):
        if self.namespace:
            name = f'{self.namespace}:{name}'

        if name in self.names:
            suffix = '$' + str(self.names[name])
            self.names[name] += 1
        else:
            suffix = ''
            self.names[name] = 1

        return name + suffix

    def register_implicit_scope(self, scope):
        self.implicit_scopes.append(scope)
        return scope.__enter__()


class StagingContext:
    '''Helper class managing context in IR staging.'''
    scope_stack: List[StagingScope] = []
    closure: Dict[str, Any] = {}

    @staticmethod
    def top():
        return StagingContext.scope_stack[-1] if len(StagingContext.scope_stack) > 0 else None

    @staticmethod
    def push(scope: StagingScope):
        StagingContext.scope_stack.append(scope)

    @staticmethod
    def pop():
        return StagingContext.scope_stack.pop()

    @staticmethod
    def fullname(name: str) -> str:
        '''Get namespace-prepended full name of given short name.'''
        return StagingContext.top().fullname(name)

    @staticmethod
    def scope(namepsace: str, allow_return: bool, borrow: bool = False):
        '''Enter a new scope with given namespace. Return object is RAII and should be used with `with`.'''
        return StagingScope(namepsace, allow_return, borrow)

    @staticmethod
    def register_implicit_scope(scope):
        return StagingContext.top().register_implicit_scope(scope)

    @staticmethod
    def reset():
        StagingContext.scope_stack.clear()
        StagingContext.closure = {}


@dataclass
class create_var:
    '''Create a IR variable. Available in python function to transform.'''
    shape: Union[Sequence, Var]
    dtype: str
    mtype: str
    atype: str = 'cache'

    def assign(self, name: str) -> Var:
        '''Customized assign behavior. Creates a VarDef with its full name.'''
        return StagingContext.register_implicit_scope(
            _VarDef(StagingContext.fullname(name), self.shape, self.dtype, self.atype, self.mtype))


def declare_var(var_name, shape, dtype, atype, mtype):
    '''Declare parameter as a variable.'''
    return StagingContext.register_implicit_scope(_VarDef(var_name, shape, dtype, atype, mtype))


def capture_var(arr: ffi.Array, name: str = 'captured') -> Var:
    name = StagingContext.fullname(name)
    StagingContext.closure[name] = arr
    return StagingContext.register_implicit_scope(_VarDef(
        name,
        arr.shape,
        arr.dtype,
        'input',
        arr.device.main_mem_type()
    ))


def assign(name: str, value):
    '''Customized assign wrapper.
    If `value` has member function `assign`, it's regarded as a customized assign behavior and
    gets executed with the assigned target variable name.
    This wrapper is used for initializing a variable.
    '''
    if hasattr(value, 'assign'):
        return value.assign(name)
    else:
        return value


class dynamic_range:
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
        with For(StagingContext.fullname(name), self.start, self.stop, self.step) as iter_var:
            with StagingContext.scope(None, False):
                body(iter_var)


def foreach(name: str, iter, body: Callable[[Any], None]) -> None:
    '''Customized foreach wrapper.
    If `value` has member function `foreach`, its regarded as a customized foreach behavior and
    used to generate code for the python for loop.
    Otherwise, we try to execute the loop as usual.
    '''
    if hasattr(iter, 'foreach'):
        iter.foreach(name, body)
    else:
        for iter_var in iter:
            body(iter_var)


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
            with StagingContext.scope(None, False):
                then_body()
        if else_body:
            with Else():
                with StagingContext.scope(None, False):
                    return else_body()


def if_then_else_expr(predicate, then_expr, else_expr):
    '''If-then-else expression staging tool.'''
    return if_then_else(predicate, then_expr, else_expr)


def return_stmt(value):
    '''Return staging tool. Only allow return in static control flow.'''
    if not StagingContext.top().allow_return:
        raise StagingError(
            'Return is only allowed in statically deterministic control flow.')
    return value


def assert_stmt(test):
    '''Assert staging tool.'''
    if isinstance(test, ffi.Expr):
        StagingContext.register_implicit_scope(Assert(test))
    else:
        assert test


def functiondef_wrapper():
    return StagingContext.scope(ctx_stack.top().get_next_nid(), True, True)


def mark_nid(nid: str):
    ctx_stack.top().set_next_nid(StagingContext.fullname(nid))


def mark_no_deps(no_deps: str):
    ctx_stack.top().add_next_no_deps(no_deps)


def mark_prefer_libs():
    ctx_stack.top().set_next_prefer_libs()


def module_helper(name: str):
    '''Helper to get an AST node with full path to given name, which should be a symbol in current module.'''
    return ast.Attribute(ast.Attribute(ast.Name('ir', ast.Load()), 'new_transformer', ast.Load()), name, ast.Load())


def call_helper(callee, *args: ast.expr, **kwargs: ast.expr):
    '''Call helper that generates a python AST Call node with given callee and arguments AST node.'''
    return ast.Call(module_helper(callee.__name__), list(args), [ast.keyword(k, w) for k, w in kwargs.items()])


def function_helper(name: str, args: Sequence[str], body: List[ast.stmt]):
    '''Function helper that generates a python AST FunctionDef node with given name, arguments name, and body.'''
    return ast.FunctionDef(
        name=name,
        args=ast.arguments(args=[], vararg=None, kwarg=None, posonlyargs=[ast.arg(a, None) for a in args],
                           defaults=[], kwonlyargs=[], kw_defaults=[]),
        body=body,
        returns=None,
        decorator_list=[])


def location_helper(new_nodes, old_node):
    if not isinstance(new_nodes, list):
        return ast.copy_location(new_nodes, old_node)
    for n in new_nodes:
        ast.copy_location(n, old_node)
    return new_nodes


class Transformer(ast.NodeTransformer):
    def visit_Expr(self, old_node: ast.Expr) -> Any:
        '''Rule:
        `declare_var(x, ...)` -> `x = declare_var(x, ...)`: x is changed from a name string to a Var
        '''
        node: ast.Expr = self.generic_visit(old_node)
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute):
                func = func.attr
            elif isinstance(func, ast.Name):
                func = func.id
            if func == 'declare_var':
                target = node.value.args[0]
                if not isinstance(target, ast.Name):
                    raise TransformError(
                        'declare_var is only allowed on top-level parameter', target)
                target = ast.Name(target.id, ast.Store())
                node = ast.Assign([target], node.value)
        elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            text = node.value.value
            if text.startswith('nid: '):
                node = ast.Expr(call_helper(
                    mark_nid, ast.Constant(text[5:], kind=None)))
            elif text.startswith('no_deps: '):
                node = ast.Expr(call_helper(
                    mark_no_deps, ast.Constant(text[9:], kind=None)))
            elif text.startswith('prefer_libs'):
                node = ast.Expr(call_helper(mark_prefer_libs))
        return location_helper(node, old_node)

    def visit_Assign(self, old_node: ast.Assign) -> ast.Assign:
        '''Rule: `lhs = rhs` -> `lhs = assign('lhs', rhs)`'''
        node = self.generic_visit(old_node)
        # FIXME: multi-assign not implemented
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            node = ast.Assign(node.targets, call_helper(
                assign, ast.Constant(node.targets[0].id), node.value))
        return location_helper(node, old_node)

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
        node = self.generic_visit(old_node)
        if isinstance(node.target, ast.Name) and len(node.orelse) == 0:
            node = [function_helper('for_body', [node.target.id], node.body),
                    ast.Expr(call_helper(foreach, ast.Constant(
                        node.target.id), node.iter, ast.Name('for_body', ast.Load())))]
        return location_helper(node, old_node)

    def visit_Call(self, old_node: ast.Call):
        '''Rule:
        `range(...)` -> `dynamic_range(...)`
        '''
        node = self.generic_visit(old_node)
        if isinstance(node.func, ast.Name):
            if node.func.id == 'range':
                node = ast.Call(module_helper(
                    dynamic_range.__name__), node.args, node.keywords)
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
        node: ast.If = self.generic_visit(old_node)
        new_node = [function_helper('then_body', [], node.body)]
        then_body = ast.Name('then_body', ast.Load())
        if node.orelse:
            new_node.append(function_helper('else_body', [], node.orelse))
            else_body = ast.Name('else_body', ast.Load())
        else:
            else_body = ast.Constant(None)
        new_node.append(ast.Expr(call_helper(
            if_then_else_stmt, node.test, then_body, else_body)))
        return location_helper(new_node, old_node)

    def visit_IfExp(self, old_node: ast.IfExp):
        '''Rule: `body if test else orelse` -> `if_then_else_expr(test, body, orelse)`'''
        node = self.generic_visit(old_node)
        node = call_helper(if_then_else_expr, node.test,
                           node.body, node.orelse)
        return location_helper(node, old_node)

    def visit_FunctionDef(self, old_node: ast.FunctionDef) -> Any:
        node: ast.FunctionDef = self.generic_visit(old_node)
        node.decorator_list = []
        old_body = node.body
        node.body = [
            ast.With(
                items=[
                    ast.withitem(context_expr=call_helper(
                        functiondef_wrapper), optional_vars=None)
                ],
                body=old_body)
        ]
        return location_helper(node, old_node)

    def visit_Assert(self, old_node: ast.Assert) -> Any:
        node: ast.Assert = self.generic_visit(old_node)
        node = ast.Expr(call_helper(assert_stmt, node.test))
        return location_helper(node, old_node)


def _remove_indent(src: str) -> str:
    lines = src.split('\n')
    spaces_to_remove = next(
        (i for i, x in enumerate(lines[0]) if x != ' '), len(lines[0]))
    return '\n'.join(line[spaces_to_remove:] for line in lines)


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


def into_staging(func, src=None):
    if src is None:
        src = _remove_indent(ins.getsource(func))
        tree = ast.parse(src)
        src, lineno = ins.getsourcelines(func)
        file = ins.getfile(func)
    else:
        tree = ast.parse(src)
        src = src.splitlines()
        lineno = 1
        file = func.__name__
    tree = ast.fix_missing_locations(Transformer().visit(tree))
    import astor
    print(astor.to_source(tree))
    caller_env = _get_caller_env(2)
    exec(compile(tree, filename='<staging>', mode='exec'), caller_env)
    return caller_env[func.__name__]


def transform(func):
    params = list(inspect.signature(func).parameters)
    staging_func = into_staging(func)

    try:
        with StagingContext.scope(None, True):
            for p in params:
                StagingContext.top().names[p] = 1
            returns = staging_func(*params)
            if isinstance(returns, Var):
                returns = [returns]
            elif isinstance(returns, tuple):
                for ret in returns:
                    if not isinstance(ret, Var):
                        raise StagingError(
                            'Illegal return at top level, need to be a `Var` or a tuple of `Var`s')
                returns = list(returns)
            elif returns is None:
                returns = []
            else:
                raise StagingError(
                    'Illegal return at top level, need to be a `Var` or a tuple of `Var`s')
            for ret in returns:
                ret.vardef.set_atype('output')
            returns = [(ret.vardef.name, ret.vardef.dtype) for ret in returns]
    finally:
        StagingContext.reset()
        staged_ast = pop_ast()

    staged = Func(func.__name__, params + list(StagingContext.closure.keys()),
                  returns, staged_ast, StagingContext.closure)

    return staged


def inline(func, src=None):
    return into_staging(func, src)
