'''
New transformer implementation based on generating a staging function.
'''

import ir
import collections

import ffi

import sys
import ast
import numpy as np
import inspect
import sourceinspect as ins
import copy
from typing import Callable, Dict, List, Sequence, Optional, Mapping, Any, Union
from dataclasses import dataclass

from . import nodes
from .nodes import (_VarDef, Var, pop_ast, For, If, Else, MarkNid, intrinsic,
                    l_and, l_or, l_not, if_then_else, ctx_stack,
                    Func, Assert)
from .utils import *

assert sys.version_info >= (3,
                            8), "Python version lower than 3.8 is not supported"

# Required for staging code to run


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


@dataclass
class StagingScope:
    '''Helper class managing scopes in staging an IR tree.'''
    appended_namespace: str
    is_static_control_flow: bool
    previous_namespace: str = None
    previous_names: Mapping[str, int] = None
    previous_is_static_control_flow: bool = None
    previous_implicit_scopes: List[_VarDef] = None

    def __enter__(self):
        '''
        Entering a new scope.
        It updates current full namespace with the new scope name, stashes previous names,
        and prepares a clean variable name environment.
        '''
        self.previous_namespace = StagingContext.namespace
        self.previous_names = StagingContext.names
        self.previous_is_static_control_flow = StagingContext.is_static_control_flow
        self.previous_implicit_scopes = StagingContext.implicit_scopes
        StagingContext.namespace += ':' + self.appended_namespace
        StagingContext.names = {}
        StagingContext.is_static_control_flow = self.is_static_control_flow
        StagingContext.implicit_scopes = []

    def __exit__(self, _1, _2, _3):
        '''
        Leaving this scope.
        It recovers names list and namespace to previous level.
        '''
        StagingContext.namespace = self.previous_namespace
        StagingContext.names = self.previous_names
        StagingContext.is_static_control_flow = self.previous_is_static_control_flow
        for vd in reversed(StagingContext.implicit_scopes):
            vd.__exit__(None, None, None)
        StagingContext.implicit_scopes = self.previous_implicit_scopes


class StagingContext:
    '''Helper class managing context in IR staging.'''
    namespace = ''
    names = {}
    is_static_control_flow = True
    implicit_scopes = []

    @staticmethod
    def fullname(name: str) -> str:
        '''Get namespace-prepended full name of given short name.'''
        if name in StagingContext.names:
            suffix = '_' + str(StagingContext.names[name])
            StagingContext.names[name] += 1
        else:
            suffix = ''
            StagingContext.names[name] = 0
        return f'{StagingContext.namespace}:{name}{suffix}'

    @staticmethod
    def scope(namepsace: str, is_static_control_flow: bool) -> StagingScope:
        '''Enter a new scope with given namespace. Return object is RAII and should be used with `with`.'''
        return StagingScope(namepsace, is_static_control_flow)

    @staticmethod
    def register_implicit_scope(scope: _VarDef):
        StagingContext.implicit_scopes.append(scope)
        return scope.__enter__()


@dataclass
class create_var:
    '''Create a IR variable. Available in python function to transform.'''
    shape: Union[Sequence, Var]
    dtype: str
    mtype: str
    atype: str = 'cache'

    def assign(self, name: str) -> _VarDef:
        '''Customized assign behavior. Creates a VarDef with its full name.'''
        return StagingContext.register_implicit_scope(
            _VarDef(StagingContext.fullname(name), self.shape, self.dtype, self.atype, self.mtype))


def declare_var(var_name, shape, dtype, atype, mtype):
    '''Declare parameter as a variable.'''
    return StagingContext.register_implicit_scope(_VarDef(var_name, shape, dtype, atype, mtype))


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
            with StagingContext.scope(f'for-{name}', False):
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
            with StagingContext.scope('if_then', False):
                then_body()
        if else_body:
            with Else():
                with StagingContext.scope('if_else', False):
                    return else_body()


def if_then_else_expr(predicate, then_expr, else_expr):
    '''If-then-else expression staging tool.'''
    return if_then_else(predicate, then_expr, else_expr)


def return_stmt(value):
    '''Return staging tool. Only allow return in static control flow.'''
    if not StagingContext.is_static_control_flow:
        raise StagingError(
            'Return is only allowed in statically deterministic control flow.')
    return value

def assert_stmt(test):
    '''Assert staging tool.'''
    if isinstance(test, ffi.Expr):
        StagingContext.register_implicit_scope(Assert(test))
    else:
        assert test


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
        return location_helper(node, old_node)

    def visit_Assign(self, old_node: ast.Assign) -> ast.Assign:
        '''Rule: `lhs = rhs` -> `lhs = assign('lhs', rhs)`'''
        node = self.generic_visit(old_node)
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
        node = self.generic_visit(old_node)
        node = [function_helper('then_body', [], node.body),
                function_helper('else_body', [], node.orelse),
                ast.Expr(call_helper(if_then_else_stmt, node.test, ast.Name(
                    'then_body', ast.Load()), ast.Name('else_body', ast.Load())))]
        return location_helper(node, old_node)

    def visit_IfExp(self, old_node: ast.IfExp):
        '''Rule: `body if test else orelse` -> `if_then_else_expr(test, body, orelse)`'''
        node = self.generic_visit(old_node)
        node = call_helper(if_then_else_expr, node.test,
                           node.body, node.orelse)
        return location_helper(node, old_node)

    def visit_Return(self, old_node: ast.Return) -> Any:
        '''Rule: `return x` -> `return return_stmt(x)`'''
        node = self.generic_visit(old_node)
        node = ast.Return(call_helper(return_stmt, node.value))
        return location_helper(node, old_node)
    
    def visit_FunctionDef(self, old_node: ast.FunctionDef) -> Any:
        node: ast.FunctionDef = self.generic_visit(old_node)
        node.decorator_list = []
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
    new_scope = {}
    exec(compile(tree, filename='<staging>', mode='exec'), globals(), new_scope)
    return new_scope[func.__name__]


def transform(func):
    params = list(inspect.signature(func).parameters)

    staging_func = into_staging(func)
    with StagingContext.scope(func.__name__, True):
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

    # TODO: handle closure
    return Func(func.__name__, params, returns, pop_ast())


def inline(func, src=None):
    return into_staging(func, src)
