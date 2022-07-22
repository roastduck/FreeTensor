'''
A staging framework to support the FreeTensor frontend.
'''
from __future__ import annotations

import abc
import re
import sys
import ast
import functools
import inspect
import traceback
import sourceinspect as ins
from typing import Callable, Dict, List, Sequence, Optional, Any, TypeVar, Union
from dataclasses import dataclass

import freetensor_ffi as ffi

assert sys.version_info >= (3,
                            8), "Python version lower than 3.8 is not supported"


class TransformError(Exception):
    '''Error occurred during AST transforming from python function to staging function
    that generates IR tree.'''

    def __init__(self, message: str, filename: str, base_lineno: int,
                 error_node: ast.AST) -> None:
        super().__init__(
            f'At {filename}:{base_lineno + error_node.lineno}:\n    {message}.')


class StagingError(Exception):
    '''Error occurred during staging function execution (i.e. IR tree generation).'''

    def __init__(self, overload: StagingOverload, message: str) -> None:
        # TODO: add output of StagingContext.call_stack
        super().__init__(
            f'{message}:\n{"".join(traceback.format_list(overload.debug_call_stack))}'
            .lstrip())


@dataclass
class AllowShortcutScope:
    '''Allow return scope.
    This is a context manager that allows return in statically deterministic control
    flow.
    '''
    overload: StagingOverload
    should_allow: bool

    def __enter__(self):
        self.prev = self.overload.is_shortcut_allowed
        self.overload.is_shortcut_allowed = self.should_allow

    def __exit__(self, exc_class, exc_value, traceback):
        self.overload.is_shortcut_allowed = self.prev


class ReturnException(Exception):
    '''Exception to be raised by StagingOverload.return_stmt.
    Holds a return value that will be passed through to the function wrapper.'''

    def __init__(self, value: Any) -> None:
        self.value = value


class BreakException(Exception):
    '''Exception to be raised by StagingOverload.break_stmt.
    Breaks from a for loop.'''
    pass


class ContinueException(Exception):
    '''Exception to be raised by StagingOverload.continue_stmt.
    Continues a for loop.'''
    pass


def _remove_indent(lines: List[str]) -> str:
    spaces_to_remove = next((i for i, x in enumerate(lines[0]) if x != ' '),
                            len(lines[0]))
    return ''.join(line[spaces_to_remove:] for line in lines)


def process_annotating_comments(src: str):
    new_src = []
    for line in src.splitlines():
        indent = re.match('\\s*', line)[0]
        rest_line = line[len(indent):]
        if rest_line.startswith('#! '):
            arg = rest_line[3:].replace('"', '\\"')
            new_src.append(f'{indent}__staging_overload__.metadata("{arg}")')
        else:
            new_src.append(line)
    new_src = '\n'.join(new_src)
    return new_src


def ast_index(idx):
    if sys.version_info < (3, 9):
        return ast.Index(idx)
    else:
        return idx


class LocalsDictWrapper:

    def __init__(self, closure: Dict[str, Any]):
        self.__dict__['closure'] = closure

    def __getattr__(self, name: str) -> Any:
        return self.__dict__['closure'][name].cell_contents

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__['closure'][name].cell_contents = value


class StagingOverload:

    def __init__(self) -> None:
        self.is_shortcut_allowed: bool = True
        self.debug_call_stack: List[traceback.FrameSummary] = []

    def custom_attr(self, obj: Any, attr: str) -> Any:
        '''
        Customized attribute accessor.

        The framework first looks for a Python native attribute. If not found, it looks
        for this overloaded custom attribute resolver.

        The default implementation provides no custom attribute.
        Can be overridden by subclasses.

        Parameters
        ----------
        obj : Any
            Object to access attribute.
        attr : str
            Attribute name.

        Returns
        -------
        Any :
            The attribute value.
        
        Throws
        ------
        AttributeError :
            If the attribute is not found.
        '''
        return None

    def metadata(self, content) -> None:
        '''
        Metadata handler.

        A metadata line is a comment starting with `#! ` and followed by a metadata,
        represented as a string parameter.

        Defaults to a no-op.
        Can be overridden by subclasses.

        Parameters
        ----------
        content : str
            The metadata content.
        '''
        pass

    def at_position(self, filename: str, lineno: int) -> None:
        '''
        Code position handler.

        Defaults to a no-op.
        Can be overridden by subclasses.

        Parameters
        ----------
        filename : str
            Name of the file containing code for the next statement.
        lineno : int
            Line number of the next statement.
        '''
        pass

    def error(self, content: str):
        return StagingError(self, content)

    def allow_shortcut_scope(self, allow: bool):
        '''Opens a scope that allows shortcut control flows in a statically deterministic
        context. Need to be closed by `with` statement.'''
        return AllowShortcutScope(self, allow)

    def foreach(self, name: str, iter, body: Callable[[Any], None]) -> None:
        '''Customized foreach wrapper.
        If `value` is instance of `StagedIterable`, its regarded as a customized foreach
        behavior and used to generate code for the python for loop.
        Otherwise, we try to execute the loop as usual.
        '''
        if isinstance(iter, StagedIterable):
            iter.foreach(name, body)
        else:
            for iter_var in iter:
                try:
                    body(iter_var)
                except BreakException:
                    break
                except ContinueException:
                    continue

    def assign_stmt(self, name: str, value):
        '''Customized assign wrapper.
        If `value` is instance of `StagedAssignable`, it's regarded as a customized
        assign behavior and gets executed with the assigned target variable name.
        This wrapper is used for initializing a variable.
        '''
        if isinstance(value, StagedAssignable):
            return value.assign(name)
        else:
            return value

    def if_then_else_stmt(self, predicate, then_body, else_body=None):
        '''If-then-else statement staging tool.
        When predicate is deterministic in staging, only one branch is generated.
        Otherwise, a If node in IR is generated.
        '''
        if isinstance(predicate, StagedPredicate):
            predicate.if_then_else_stmt(then_body, else_body)
        else:
            if predicate:
                then_body()
            elif else_body:
                else_body()

    def if_then_else_expr(self, predicate, then_expr, else_expr):
        '''If-then-else expression staging tool.'''
        if isinstance(predicate, StagedPredicate):
            return predicate.if_then_else_expr(then_expr, else_expr)
        else:
            if predicate:
                return then_expr()
            else:
                return else_expr()

    def while_stmt(self, fpred, body):
        '''While statement staging tool.'''
        first_pred = fpred()
        if isinstance(first_pred, StagedPredicate):
            first_pred.while_stmt(body)
        else:
            if first_pred:
                body()
            while fpred():
                body()

    def assert_stmt(self, test):
        '''Assert staging tool.'''
        if isinstance(test, StagedPredicate):
            test.assert_stmt()
        else:
            assert test

    def return_stmt(self, value, funcname):
        '''Return staging tool. Only allow return in static control flow.'''
        if not self.is_shortcut_allowed:
            raise self.error(
                'Return is only allowed in statically deterministic control flow.'
            )
        if isinstance(value, StagedAssignable):
            value = value.assign(funcname)
        raise ReturnException(value)

    def break_stmt(self):
        '''Break staging tool. Only allow break in static control flow.'''
        if not self.is_shortcut_allowed:
            raise self.error(
                'Break is only allowed in statically deterministic control flow.'
            )
        raise BreakException()

    def continue_stmt(self):
        '''Continue staging tool. Only allow continue in static control flow.'''
        if not self.is_shortcut_allowed:
            raise self.error(
                'Continue is only allowed in statically deterministic control flow.'
            )
        raise ContinueException()

    def load_attr(self, obj, attr: str):
        '''Load attribute staging tool. Allows customization of reading attributes.'''
        try:
            return getattr(obj, attr)
        except AttributeError:
            try:
                # Have to use AttributeError again, since a custom attribute might have
                # a None value
                result = self.custom_attr(obj, attr)
                successful = True
            except AttributeError:
                successful = False

            if successful:
                return result
            else:
                raise

    def and_expr(self, *lazy_args):

        def reducer(a, fb):
            if isinstance(a, StagedPredicate):
                return a.logical_and(fb)
            else:
                # This is not a simple logical and; it's equivalent to a if-then-else.
                # Thus, if a is True, fb() is returned, preserving the original value,
                # which might be a StagedPredicate.
                return a and fb()

        return functools.reduce(reducer, lazy_args, True)

    def or_expr(self, *lazy_args):

        def reducer(a, fb):
            if isinstance(a, StagedPredicate):
                return a.logical_or(fb)
            else:
                return a or fb()

        return functools.reduce(reducer, lazy_args, False)

    def not_expr(self, arg):
        if isinstance(arg, StagedPredicate):
            return arg.logical_not()
        else:
            return not arg

    def functiondef_decorator(self, filename):
        return functools.partial(self.functiondef_wrapper, filename)

    def functiondef_wrapper(self, filename, func):
        '''Function definition wrapper.
        This wrapper performs extra initialization and cleanup for function definition.
        '''

        def wrapped(*args, **kwargs):
            # Push debug call stack with some random line number.
            # It will be updated by `mark_position` calls in the function.
            self.debug_call_stack.append(
                traceback.FrameSummary(filename, 1, func.__name__))
            # The called function can now return from itself, despite what the outer
            # control flow is.
            with self.allow_shortcut_scope(True):
                try:
                    func(*args, **kwargs)
                except ReturnException as e:
                    result = e.value
                else:
                    # No return_stmt was called, naturally returns None
                    result = None
            # Pop debug call stack.
            self.debug_call_stack.pop()
            return result

        return wrapped

    def annotate_stmt(self, name: str, ty):
        if isinstance(ty, StagedTypeAnnotation):
            return ty.annotate(name)
        return None

    def mark_position(self, base_lineno: int, line_offset: int):
        # FrameSummary is immutable, so we have to initialize a new one with updated
        # line number.
        self.debug_call_stack[-1] = traceback.FrameSummary(
            self.debug_call_stack[-1].filename, base_lineno + line_offset - 1,
            self.debug_call_stack[-1].name)
        self.at_position(self.debug_call_stack[-1].filename,
                         self.debug_call_stack[-1].lineno)

    def into_staging(self,
                     func,
                     extra_locals: Dict[str, Any] = {},
                     src: str = None,
                     verbose=False):
        assert inspect.isfunction(func)

        if src is None:
            lines, lineno = ins.getsourcelines(func)
            src = _remove_indent(lines)
            file = ins.getfile(func)
        else:
            lineno = 1
            file = f'<staging:{func.__name__}>'

        # Inject overload to extra_locals.
        extra_locals['__staging_overload__'] = self

        # To transform a function, except essential AST transformation, we have to pass
        # the globals and locals (actually captured outer local variables) to the
        # transformed function properly.
        # Note that:
        # 1. We have to pass both globals and locals to `exec`.
        # 2. We cannot insert locals to the globals `dict`, otherwise it will pollute
        #    the globals `dict`.
        # 3. We cannot copy the globals `dict` before passing it to exec, otherwise the
        #    staged function cannot write to globals and get later updates in the global.
        # Thus, we have to pass the globals and locals to the transformed function
        # separately.

        if func.__closure__:
            assert len(func.__code__.co_freevars) == len(func.__closure__)
            func_locals = {
                name: cell for name, cell in zip(func.__code__.co_freevars,
                                                 func.__closure__)
            }
        else:
            func_locals = {}

        tree = ast.parse(process_annotating_comments(src))
        tree = Transformer(file, lineno).visit(tree)

        # Instead of passing the `func_local` directly to `exec`, we instead wrap the
        # staging function. This is to workaround an issue of CPython. (See
        # https://github.com/python/cpython/issues/86084).

        # The sketch is:
        # ```
        # def __freetensor_staging_wrapper__(__freetensor_extra_locals__,
        #                                    __freetensor_local_cells__):
        #     some_extra_local = __freetensor_extra_locals__['some_extra_local']
        #     some_captured = None
        #
        #     def original_func():
        #         nonlocal some_captured
        #         some_captured = __freetensor_local_cells__.some_captured
        #         try:
        #             ...  # original function body
        #         finally:
        #             __freetensor_local_cells__.some_captured = some_captured
        #
        #     return original_func
        # ```
        # Note that `__freetensor_local_cells__` is a `LocalsDictWrapper` object.
        # It in-turn accesses cell.cell_contents to get/set the value of the local
        # variable.
        # The `LocalsDictWrapper` is a helper class to reduce code generation complexity.

        WRAPPER_NAME = '__freetensor_staging_wrapper__'
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)

        # Modify function body.
        if len(func_locals) > 0:
            tree.body[0].body = ([
                # Declare them as nonlocals to assign to outer scope.
                ast.Nonlocal(list(func_locals.keys())),
            ] + [
                # Fetch latest values of the closure variables.
                ast.Assign([ast.Name(name, ast.Store())],
                           ast.Attribute(
                               ast.Name('__freetensor_local_cells__',
                                        ast.Load()), name, ast.Load()))
                for name in func_locals.keys()
            ] + [
                # Use a try-finally to ensure closure write back.
                ast.Try(body=tree.body[0].body,
                        handlers=[],
                        orelse=[],
                        finalbody=[
                            ast.Assign([
                                ast.Attribute(
                                    ast.Name('__freetensor_local_cells__',
                                             ast.Load()), name, ast.Store())
                            ], ast.Name(name, ast.Load()))
                            for name in func_locals.keys()
                        ])
            ])
        tree.body = [
            ast.FunctionDef(
                name=WRAPPER_NAME,
                args=ast.arguments(posonlyargs=[],
                                   args=[
                                       ast.arg('__freetensor_extra_locals__',
                                               None),
                                       ast.arg('__freetensor_local_cells__',
                                               None)
                                   ],
                                   vararg=None,
                                   kwonlyargs=[],
                                   kw_defaults=[],
                                   kwarg=None,
                                   defaults=[]),
                body=[
                    # Captured closure variables are not fetched here, only declared.
                    ast.Assign([ast.Name(name, ast.Store())],
                               ast.Constant(None))
                    for name in func_locals.keys()
                ] + [
                    # Extra locals are fetched here.
                    ast.Assign([ast.Name(name, ast.Store())],
                               ast.Subscript(
                                   ast.Name('__freetensor_extra_locals__',
                                            ast.Load()),
                                   ast_index(ast.Constant(name)), ast.Load()))
                    for name in extra_locals.keys()
                ] + tree.body +
                [ast.Return(value=ast.Name(id=func.__name__, ctx=ast.Load()))],
                decorator_list=[],
                returns=None),
        ]
        tree = ast.fix_missing_locations(tree)

        if verbose:
            import astor
            source = astor.to_source(tree)

            from pygments import highlight
            from pygments.lexers import PythonLexer
            from pygments.formatters import TerminalFormatter
            print(highlight(source, PythonLexer(),
                            TerminalFormatter(bg='dark', linenos=True)),
                  file=sys.stderr)

            tree = source  # make debug info match dumped source

        # Create an empty locals dict to avoid polluting the original globals.
        empty_locals = {}
        exec(compile(tree, f'<staging:{func.__name__}>', 'exec'),
             func.__globals__, empty_locals)
        f_wrapper = empty_locals[WRAPPER_NAME]
        # Pass the closure to the wrapper and retrieve the staging function with
        # correct captured variables.
        f_staging = f_wrapper(extra_locals, LocalsDictWrapper(func_locals))

        return f_staging


class StagedIterable:

    def foreach(self, name: str, f: Callable[[Any], None]):
        raise NotImplementedError()


class StagedAssignable(abc.ABC):

    @abc.abstractmethod
    def assign(self, name: str):
        raise NotImplementedError()


class StagedTypeAnnotationMeta(abc.ABCMeta):

    def __getitem__(self, args):
        return self(*args)


class StagedTypeAnnotation(metaclass=StagedTypeAnnotationMeta):

    @abc.abstractmethod
    def annotate(self, name: str):
        raise NotImplementedError()


class StagedPredicate(abc.ABC):

    @abc.abstractmethod
    def logical_and(
            self, lazy_other: Callable[[], StagedPredicate]) -> StagedPredicate:
        raise NotImplementedError()

    @abc.abstractmethod
    def logical_or(
            self, lazy_other: Callable[[], StagedPredicate]) -> StagedPredicate:
        raise NotImplementedError()

    @abc.abstractmethod
    def logical_not(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def if_then_else_stmt(self, then_body: Callable[[], None],
                          else_body: Optional[Callable[[], None]]):
        raise NotImplementedError()

    @abc.abstractmethod
    def if_then_else_expr(self, then_expr: Callable[[], Any],
                          else_expr: Callable[[], Any]):
        raise NotImplementedError()

    @abc.abstractmethod
    def while_stmt(self, body: Callable[[], None]):
        raise NotImplementedError()

    @abc.abstractmethod
    def assert_stmt(self):
        raise NotImplementedError()


def call_helper(callee, *args: ast.expr, **kwargs: ast.expr):
    '''Call helper that generates a python AST Call node with given callee (overload
    member) and arguments AST node.'''
    return ast.Call(
        ast.Attribute(ast.Name('__staging_overload__',
                               ast.Load()), callee.__name__, ast.Load()),
        list(args), [ast.keyword(k, w) for k, w in kwargs.items()])


def function_helper(name: str, args: Sequence[str], body: List[ast.stmt],
                    nonlocals: List[str]):
    '''Function helper that generates a python AST FunctionDef node with given name,
    arguments name, and body.'''
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


_EMPTY_ARGS = ast.arguments(args=[],
                            vararg=None,
                            kwarg=None,
                            posonlyargs=[],
                            defaults=[],
                            kwonlyargs=[],
                            kw_defaults=[])


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
                    call_helper(StagingOverload.mark_position,
                                ast.Constant(self.base_lineno),
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
                    call_helper(StagingOverload.assign_stmt, ast.Constant(name),
                                node.value))
        return location_helper(node, old_node)

    def handleType_AnnAssign(self, node: ast.AnnAssign) -> Any:
        x = node.target
        assert isinstance(x, ast.Name)
        assert node.value is None
        x_str = ast.Constant(x.id)
        Ty = node.annotation

        intermediate = f'freetensor__annotate__{x.id}'
        intermediate_store = ast.Name(intermediate, ast.Store())
        intermediate_load = ast.Name(intermediate, ast.Load())
        node = [
            ast.Assign([intermediate_store],
                       call_helper(StagingOverload.annotate_stmt, x_str, Ty)),
            ast.If(intermediate_load, [ast.Assign([x], intermediate_load)], [])
        ]

        return node

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
            node = self.handleType_AnnAssign(node)
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
                # While opening a fake function, For loops initiates an iter name as
                # well. Need to remove it from the outer nonlocals list to implement
                # shadowing. Only For loops behaves as such, so handle it specially here.
                nonlocals = set(nonlocals)
                if old_node.target.id in nonlocals:
                    nonlocals.remove(old_node.target.id)
                nonlocals = list(nonlocals)

                node = self.generic_visit(old_node)
                node = [
                    function_helper('for_body', [node.target.id], node.body,
                                    nonlocals),
                    ast.Expr(
                        call_helper(StagingOverload.foreach,
                                    ast.Constant(node.target.id), node.iter,
                                    ast.Name('for_body', ast.Load())))
                ]
        else:
            node = self.generic_visit(old_node)
        return location_helper(node, old_node)

    def visit_While(self, old_node: ast.While) -> Any:
        '''Rule:
        ```
        while pred:
            body
        ```
        ->
        ```
        def while_body():
            body
        while_stmt(lambda: pred, while_body)
        ```'''
        with NonlocalTransformingScope(self) as nonlocals:
            node: ast.While = self.generic_visit(old_node)
            node = [
                function_helper('while_body', [], node.body, nonlocals),
                ast.Expr(
                    call_helper(StagingOverload.while_stmt,
                                ast.Lambda(_EMPTY_ARGS, node.test),
                                ast.Name('while_body', ast.Load())))
            ]
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
            ast.Expr(
                call_helper(StagingOverload.if_then_else_stmt, test, then_body,
                            else_body)))
        return location_helper(new_node, old_node)

    def visit_IfExp(self, old_node: ast.IfExp):
        '''Rule: `body if test else orelse` -> `if_then_else_expr(test, body, orelse)`'''
        node = self.generic_visit(old_node)
        node = call_helper(StagingOverload.if_then_else_expr, node.test,
                           ast.Lambda(_EMPTY_ARGS, node.body),
                           ast.Lambda(_EMPTY_ARGS, node.orelse))
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

            # Transform the function body
            node: ast.FunctionDef = self.generic_visit(old_node)
            # Cleanup the decorators
            node.decorator_list = [
                call_helper(StagingOverload.functiondef_decorator,
                            ast.Constant(self.filename))
            ]
            # Handle the type annotations
            node.body = [
                stmt for arg in node.args.posonlyargs + node.args.args
                if arg.annotation for stmt in self.handleType_AnnAssign(
                    location_helper(
                        ast.AnnAssign(ast.Name(arg.arg, ast.Store()),
                                      arg.annotation, None, 1), old_node))
            ] + node.body

            # Cleanup annotations; we don't need them anymore
            for arg in [
                    node.args.vararg, node.args.kwarg
            ] + node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                if arg is not None:
                    arg.annotation = None

        self.curr_func = prev_func
        self.nonlocals = prev_nonlocals
        return location_helper(node, old_node)

    def visit_Assert(self, old_node: ast.Assert) -> Any:
        node: ast.Assert = self.generic_visit(old_node)
        node = ast.Expr(call_helper(StagingOverload.assert_stmt, node.test))
        return location_helper(node, old_node)

    def visit_BoolOp(self, old_node: ast.BoolOp) -> Any:
        node: ast.BoolOp = self.generic_visit(old_node)
        if isinstance(node.op, ast.And):
            libfunc = StagingOverload.and_expr
        elif isinstance(node.op, ast.Or):
            libfunc = StagingOverload.or_expr
        else:
            return location_helper(node, old_node)
        node = call_helper(libfunc,
                           *[ast.Lambda(_EMPTY_ARGS, v) for v in node.values])
        return location_helper(node, old_node)

    def visit_UnaryOp(self, old_node: ast.UnaryOp) -> Any:
        node: ast.UnaryOp = self.generic_visit(old_node)
        if isinstance(node.op, ast.Not):
            node = call_helper(StagingOverload.not_expr, node.operand)
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

    def visit_Attribute(self, old_node: ast.Attribute) -> Any:
        node: ast.Attribute = self.generic_visit(old_node)
        if isinstance(node.ctx, ast.Load):
            if not (isinstance(node.value, ast.Name) and
                    node.value.id == '__staging_overload__'):
                node = call_helper(StagingOverload.load_attr, node.value,
                                   ast.Constant(node.attr))
        return location_helper(node, old_node)

    def visit_Return(self, old_node: ast.Return) -> Any:
        node: ast.Return = self.generic_visit(old_node)
        assert self.curr_func is not None
        node = ast.Expr(
            call_helper(StagingOverload.return_stmt, node.value,
                        ast.Constant(self.curr_func)))
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

    def visit_Yield(self, node: ast.Yield) -> Any:
        raise NotImplementedError()

    def visit_YieldFrom(self, node: ast.YieldFrom) -> Any:
        raise NotImplementedError()

    def visit_Break(self, node: ast.Break) -> Any:
        return ast.Expr(call_helper(StagingOverload.break_stmt))

    def visit_Continue(self, node: ast.Continue) -> Any:
        return ast.Expr(call_helper(StagingOverload.continue_stmt))
