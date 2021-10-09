import collections
import ffi

import sys
import ast
import numpy as np
import inspect
import sourceinspect as ins
from typing import Sequence, Optional, Mapping, Any

from . import nodes
from .nodes import (_VarDef, Var, pop_ast, For, If, Else, MarkNid, intrinsic,
                    l_and, l_or, l_not, if_then_else, ctx_stack as node_ctx,
                    Func, Tensor)
from .utils import *

assert sys.version_info >= (3,
                            8), "Python version lower than 3.8 is not supported"


def declare_var(var, shape, dtype, atype, mtype, name=None):
    pass


def create_var(shape, dtype, atype, mtype, name=None):
    return np.zeros(shape, dtype)


class ASTContext:

    def __init__(self):
        self.vardef_stack = []
        self.var_dict = {}
        self.old_vars = []


class ASTContextStack:

    def __init__(self):
        self.ctx_stack = []
        self.name_map = {}
        self.next_nid = ""
        self.node_ctx_bak = node_ctx.get_stack()
        self.now_context_id = 0
        node_ctx.reset()

    def __del__(self):
        node_ctx.set_stack(self.node_ctx_bak)

    def top(self) -> ASTContext:
        return self.ctx_stack[-1]

    def get_current_name(self, name):
        return self.name_map.get(name)

    def create_current_name(self, name, atype, override_name=None):
        if override_name is not None:
            if override_name in self.name_map:
                raise ffi.InvalidProgram(f"Name {override_name} already exists")
            self.name_map[name] = override_name
            return override_name

        modify_name = lambda old, i: "$" + old + (""
                                                  if i == 0 else "_" + str(i))
        name_id = 0
        while modify_name(name, name_id) in self.name_map:
            name_id += 1
        new_name = modify_name(name, name_id)
        self.name_map[name] = new_name
        return new_name

    def find_var_by_name(self, name) -> Optional[Var]:
        name = self.get_current_name(name)

        for ctx in reversed(self.ctx_stack):  # type: ASTContext
            if name in ctx.old_vars:
                assert False, f"Variable {name} reassigned in if/for/while"
            var = ctx.var_dict.get(name)
            if var is not None:
                return var

        return None

    def create_scope(self):
        self.ctx_stack.append(ASTContext())

    def pop_scope(self):
        assert self.ctx_stack, "Bug: scope stack is empty when pop_scope"
        popped = self.ctx_stack.pop()  # type: ASTContext
        if self.ctx_stack:
            top = self.top()
            top.old_vars.extend(popped.var_dict.keys())
        for var in reversed(popped.vardef_stack):  # type: _VarDef
            var.__exit__(None, None, None)

    def create_variable(self,
                        name,
                        shape,
                        dtype,
                        atype,
                        mtype,
                        override_name=None):
        name = self.create_current_name(name,
                                        atype,
                                        override_name=override_name)
        vardef = _VarDef(name, shape, dtype, atype, mtype)
        var = vardef.__enter__()
        top = self.top()
        top.vardef_stack.append(vardef)
        top.var_dict[name] = var
        return var

    def create_loop(self, name, begin, end):
        name = self.create_current_name(name, "cache")
        fr = For(name, begin, end, self.get_nid(), self.get_no_deps())
        var = fr.__enter__()
        top = self.top()
        top.var_dict[name] = var
        return fr

    def set_nid(self, name: str):
        MarkNid(name)

    def get_nid(self):
        ret = node_ctx.top().get_next_nid()
        MarkNid("")
        return ret

    def set_no_deps(self):
        node_ctx.top().set_next_no_deps(True)

    def get_no_deps(self):
        ret = node_ctx.top().get_next_no_deps()
        node_ctx.top().set_next_no_deps(False)
        return ret

    def new_context_id(self):
        self.now_context_id += 1
        return self.now_context_id


class VarCreation:

    def __init__(self,
                 ctx_stack: ASTContextStack,
                 shape: Sequence,
                 dtype,
                 atype,
                 mtype,
                 name=None,
                 override_name=None):
        self.ctx_stack = ctx_stack
        self.shape = shape
        self.dtype = dtype
        self.atype = atype
        self.mtype = mtype
        self.name = name
        self.override_name = override_name

    def add_name(self, name):
        assert self.name is None, "Bug: Variable name is set more than once"
        self.name = name

    def execute(self):
        assert self.name is not None, "Bug: Variable name is not set"
        return self.ctx_stack.create_variable(self.name,
                                              self.shape,
                                              self.dtype,
                                              self.atype,
                                              self.mtype,
                                              override_name=self.override_name)


class InlineFunction:

    def __init__(self, name, body, params, globals):
        self.name = name
        self.body = body
        self.params = params
        self.globals = globals
        self.arg_var = None

    def set_arg_var(self, arg_var):
        self.arg_var = arg_var

    def expand(self, ctx_stack, nid):
        assert self.arg_var is not None
        transformer = ASTTransformer(ctx_stack, self.params, self.globals)
        transformer.set_replace(self.arg_var, nid)
        for stmt in self.body:
            transformer.visit(stmt)
        return transformer.returns


class ASTTransformer(ast.NodeTransformer):

    def __init__(self, ctx_stack: ASTContextStack, params: Sequence[str],
                 globals: Mapping[str, Any]):
        super().__init__()
        self.ctx_stack = ctx_stack
        self.params = params
        self.globals = globals
        self.allow_undefined = False
        self.replace = {}
        self.arg_var = {}
        self.created_vars = set()
        self.returned = False
        self.returns = []
        self.prefix = ""
        self.nid = ""

    def set_replace(self, arg_var, prefix):
        self.arg_var = arg_var
        if prefix:
            self.prefix = prefix
        else:
            self.prefix = '#' + str(self.ctx_stack.new_context_id())

    @staticmethod
    def parse_stmt(stmt):
        return ast.parse(stmt).body[0]

    @staticmethod
    def parse_expr(expr):
        return ast.parse(expr).body[0].value

    def get_name(self, name):
        if name in self.replace:
            while name in self.replace:
                name = self.replace[name]
        elif name in self.created_vars:
            name = self.prefix + ':' + name
        return name

    def visit_Name(self, node):
        if node.id in self.arg_var:
            node.expr_ptr = self.arg_var[node.id]
            return node
        name = self.get_name(node.id)
        var = self.ctx_stack.find_var_by_name(name)
        if var is None:
            if node.id in self.globals:
                var = self.globals[node.id]  # self.globals[node.id] can be None
            else:
                assert self.allow_undefined, f"Variable {node.id} (a.k.a {name}) used without declaration or creation"
        node.expr_ptr = var
        return node

    def visit_Constant(self, node):
        node.expr_ptr = node.value
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value.expr_ptr, Var) and node.attr == "shape":
            node.expr_ptr = lambda idx: node.value.expr_ptr.shape(idx)
        elif isinstance(node.value.expr_ptr, Var) and node.attr == "ndim":
            node.expr_ptr = node.value.expr_ptr.ndim
        elif isinstance(node.value.expr_ptr, Var) and node.attr == "dtype":
            node.expr_ptr = node.value.expr_ptr.dtype
        elif isinstance(node.value.expr_ptr, Var) and node.attr == "mtype":
            node.expr_ptr = node.value.expr_ptr.mtype
        else:
            node.expr_ptr = getattr(node.value.expr_ptr, node.attr)
        return node

    def visit_Slice(self, node):
        self.generic_visit(node)
        start = node.lower.expr_ptr if getattr(node, "lower",
                                               None) is not None else None
        stop = node.upper.expr_ptr if getattr(node, "upper",
                                              None) is not None else None
        step = node.step.expr_ptr if getattr(node, "step",
                                             None) is not None else None
        node.expr_ptr = slice(start, stop, step)
        return node

    def visit_Index(self, node):
        self.generic_visit(node)
        node.expr_ptr = node.value.expr_ptr
        return node

    def visit_ExtSlice(self, node):
        self.generic_visit(node)
        node.expr_ptr = tuple(map(lambda x: x.expr_ptr, node.dims))
        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        var = node.value.expr_ptr
        sub = node.slice.expr_ptr
        assert var is not None
        node.expr_ptr = var[sub]
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        assert hasattr(node.left, "expr_ptr"), "left operand is not expression"
        assert hasattr(node.right,
                       "expr_ptr"), "right operand is not expression"
        op = {
            ast.Add: lambda l, r: l + r,
            ast.Sub: lambda l, r: l - r,
            ast.Mult: lambda l, r: l * r,
            ast.Div: lambda l, r: l / r,
            ast.FloorDiv: lambda l, r: l // r,
            ast.Mod: lambda l, r: l % r,
        }.get(type(node.op))
        assert op is not None, "Binary operator not implemented"
        node.expr_ptr = op(node.left.expr_ptr, node.right.expr_ptr)
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        for i in node.values:
            assert hasattr(i, "expr_ptr"), "Bool operand is not expression"
        assert len(
            node.values) > 1, "Bug: Bool operator has less than one operand"
        op = {
            ast.And: lambda l, r: l_and(l, r),
            ast.Or: lambda l, r: l_or(l, r),
        }.get(type(node.op))
        assert op is not None, "Bool operator not implemented"
        expr = op(node.values[0].expr_ptr, node.values[1].expr_ptr)
        for i in node.values[2:]:
            expr = op(expr, i.expr_ptr)
        node.expr_ptr = expr
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        assert hasattr(node.operand,
                       "expr_ptr"), "Unary operand is not expression"
        op = {
            ast.Not: lambda l: l_not(l),
            ast.USub: lambda l: 0 - l,
        }.get(type(node.op))
        assert op is not None, "Unary operator not implemented"
        node.expr_ptr = op(node.operand.expr_ptr)
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        node.expr_ptr = if_then_else(node.test.expr_ptr, node.body.expr_ptr,
                                     node.orelse.expr_ptr)
        return node

    def visit_FunctionDef(self, node):
        self.ctx_stack.create_scope()
        self.generic_visit(node)
        self.ctx_stack.pop_scope()
        return node

    def get_shape(self, data, shape=None):
        if shape is None:
            shape = []
        if isinstance(data, collections.abc.Sequence):
            shape.append(len(data))
            if len(data):
                self.get_shape(data[0], shape)
        return shape

    def visit_Call(self, node):

        self.visit(node.func)
        callee = node.func.expr_ptr

        args = []
        kws = {}
        if callee is declare_var:
            assert len(
                node.args) == 5, "declare_var function requries 5 arguments"
            assert isinstance(node.args[0], ast.Name)
            self.allow_undefined = True
            self.visit(node.args[0])
            self.allow_undefined = False
            args.append(node.args[0].id)
            for i in range(1, 5):
                self.visit(node.args[i])
                args.append(node.args[i].expr_ptr)
        else:
            for arg in node.args:
                self.visit(arg)
                args.append(arg.expr_ptr)
        for item in node.keywords:
            self.visit(item)
            kws[item.arg] = item.value.expr_ptr

        if callee is create_var:
            shape, dtype, atype, mtype = args

            override_name = kws.get("name")
            node.expr_ptr = VarCreation(self.ctx_stack,
                                        shape,
                                        dtype,
                                        atype,
                                        mtype,
                                        override_name=override_name)
        elif callee is declare_var:
            name, shape, dtype, atype, mtype = args
            override_name = kws.get("name")
            nid = name
            if self.prefix:
                nid = self.prefix + ':' + nid
            MarkNid(nid)
            if self.prefix:
                name = self.get_name(name)
                var = self.ctx_stack.find_var_by_name(name)
            else:
                assert name in self.params, f"Parameter {name} not found"
                VarCreation(self.ctx_stack,
                            shape,
                            dtype,
                            atype,
                            mtype,
                            name,
                            override_name=name).execute()
        elif callee is MarkNid:
            nid, = args
            self.ctx_stack.set_nid(nid)
        elif callee is nodes.min:
            lhs, rhs = args
            node.expr_ptr = nodes.min(lhs, rhs)
        elif callee is nodes.max:
            lhs, rhs = args
            node.expr_ptr = nodes.max(lhs, rhs)
        elif callee is nodes.abs:
            expr, = args
            node.expr_ptr = nodes.abs(expr)
        elif callee is nodes.sqrt:
            expr, = args
            node.expr_ptr = nodes.sqrt(expr)
        elif callee is nodes.exp:
            expr, = args
            node.expr_ptr = nodes.exp(expr)
        elif callee is intrinsic:
            fmt_str = args[0]
            expr_args = args[1:]
            ret_type = ffi.DataType.Void
            if "ret_type" in kws:
                ret_type = parseDType(kws["ret_type"])
            node.expr_ptr = ffi.makeIntrinsic(fmt_str, expr_args, ret_type)
        elif isinstance(callee, ffi.Func):
            raise ffi.InvalidProgram("Please use @ir.inline for subroutines")
        elif isinstance(callee, InlineFunction):
            if len(args) != len(callee.params):
                raise ffi.InvalidProgram(
                    f"Number of arguments does not match when calling {callee.name}, {len(callee.params)} needed, but {len(args)} provided"
                )

            nid = '#' + str(self.ctx_stack.new_context_id())
            arg_var = {}
            for arg, param in zip(args, callee.params):
                if self.nid:
                    name = self.nid + ":" + param
                else:
                    name = nid + ":" + param
                MarkNid(name)
                if isinstance(arg, Var):
                    arg_var[param] = arg
                elif isinstance(arg, InlineFunction):
                    # only support single return value now
                    returns = arg.expand(self.ctx_stack, name)
                    assert len(returns) == 1, "only support single return value"
                    arg_var[param] = self.ctx_stack.find_var_by_name(returns[0])
                elif isinstance(arg, Tensor):
                    var = VarCreation(self.ctx_stack, arg.shape(), arg.dtype(),
                                      "cache", arg.mtype, name).execute()
                    arg_var[param] = var
                    for i in range(arg.size()):
                        var[arg.indices(i)] = arg.at(i)
                else:
                    assert False, "Invalid function argument"
            callee.set_arg_var(arg_var)
            node.expr_ptr = callee
        else:
            node.expr_ptr = callee(*args, **kws)
        return node

    def visit_Tuple(self, node):
        self.generic_visit(node)
        tup = []
        for i in node.elts:
            assert hasattr(i, "expr_ptr"), "Invalid tuple"
            tup.append(i.expr_ptr)
        tup = tuple(tup)
        node.expr_ptr = tup
        return node

    def visit_List(self, node):
        self.generic_visit(node)
        lst = []
        for i in node.elts:
            assert hasattr(i, "expr_ptr"), "Invalid list"
            lst.append(i.expr_ptr)
        node.expr_ptr = lst
        return node

    def visit_Assign(self, node):
        self.nid = self.ctx_stack.get_nid()
        self.allow_undefined = True
        for tgt in node.targets:
            self.visit(tgt)
        self.allow_undefined = False
        self.visit(node.value)

        # TODO: (maybe) support for multiple assignment
        assert len(node.targets) == 1, "Multiple assignment is not supported"
        for target in node.targets:
            assert hasattr(
                target,
                "expr_ptr"), "Target to be assigned is not an expression"
        assert hasattr(node.value,
                       "expr_ptr"), "Value to be assigned is not an expression"
        if isinstance(node.value.expr_ptr, InlineFunction):
            targets = []
            if isinstance(node.targets[0], ast.Tuple):
                for target in node.targets[0].elts:
                    assert isinstance(target, ast.Name), "Target must be a name"
                    # FIXME
                    # assert target.expr_ptr is None, f"Variable {target.id} already exists"
                    targets.append(target.id)
            else:
                assert isinstance(node.targets[0],
                                  ast.Name), "Target must be a name"
                # FIXME
                # assert node.targets[
                #     0].expr_ptr is None, f"Variable {node.targets[0].id} already exists"
                targets.append(node.targets[0].id)
            returns = node.value.expr_ptr.expand(self.ctx_stack, self.nid)
            for target, ret in zip(targets, returns):
                self.replace[target] = ret
        elif isinstance(node.value.expr_ptr, VarCreation):
            self.created_vars.add(node.targets[0].id)
            name = self.get_name(node.targets[0].id)
            nid = name
            MarkNid(nid)
            var_creation = node.value.expr_ptr
            var_creation.atype = 'cache'
            var_creation.add_name(name)
            var_creation.execute()
        elif isinstance(node.targets[0], ast.Subscript):
            MarkNid(self.nid)
            var = node.targets[0].value.expr_ptr
            sub = node.targets[0].slice.expr_ptr
            var[sub] = node.value.expr_ptr
        else:
            assert False, "Invalid assignment"
        return node

    def visit_AugAssign(self, node):
        import copy

        target_load = copy.copy(node.target)
        target_load.ctx = ast.Load()
        target_load = self.visit(target_load)

        self.generic_visit(node)
        assert hasattr(node.target,
                       "expr_ptr"), "Target to be assigned is not an expression"
        assert hasattr(node.value,
                       "expr_ptr"), "Value to be assigned is not an expression"
        if isinstance(node.target, ast.Subscript):
            op = {
                ast.Add: lambda l, r: l + r,
                ast.Sub: lambda l, r: l - r,
                ast.Mult: lambda l, r: l * r,
                ast.Div: lambda l, r: l / r,
                ast.FloorDiv: lambda l, r: l // r,
                ast.Mod: lambda l, r: l % r,
            }.get(type(node.op))
            var = node.target.value.expr_ptr
            sub = node.target.slice.expr_ptr
            var[sub] = op(node.target.expr_ptr, node.value.expr_ptr)
        else:
            assert False, "Invalid augmented assignment"
        return node

    def visit_For(self, node):
        if (isinstance(node.iter, ast.Call) and
                isinstance(node.iter.func, ast.Name) and
                node.iter.func.id == "range" and len(node.iter.args) > 0 and
                len(node.iter.args) <= 2):

            self.ctx_stack.create_scope()
            name = node.target.id
            if self.prefix:
                self.created_vars.add(name)
                name = self.prefix + ':' + name
            if len(node.iter.args) == 1:
                self.visit(node.iter.args[0])
                assert hasattr(node.iter.args[0],
                               "expr_ptr"), "For range is not expression"
                begin = 0
                end = node.iter.args[0].expr_ptr
            else:
                self.visit(node.iter.args[0])
                self.visit(node.iter.args[1])
                assert hasattr(node.iter.args[0],
                               "expr_ptr"), "For range is not expression"
                assert hasattr(node.iter.args[1],
                               "expr_ptr"), "For range is not expression"
                begin = node.iter.args[0].expr_ptr
                end = node.iter.args[1].expr_ptr
            fr = self.ctx_stack.create_loop(name, begin, end)
            for i in node.body:
                self.visit(i)
            self.ctx_stack.pop_scope()
            fr.__exit__(None, None, None)
        else:
            assert False, "For statement other than range(a, b) is not implemented"
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str):
            s = node.value.value
            if s[0:5] == "nid: ":
                name = s[5:]
                if self.prefix:
                    name = self.prefix + ':' + name
                self.ctx_stack.set_nid(name)
            if s == "no_deps":
                self.ctx_stack.set_no_deps()
            return node

        self.nid = self.ctx_stack.get_nid()
        self.generic_visit(node)
        if hasattr(node.value, "expr_ptr") and isinstance(
                node.value.expr_ptr, InlineFunction):
            node.value.expr_ptr.expand(self.ctx_stack, self.nid)
        return node

    def visit_If(self, node):
        self.visit(node.test)
        assert hasattr(node.test,
                       "expr_ptr"), "If condition is not an expression"

        # static conditions allow illegal accesses in some branches
        if node.test.expr_ptr is True:
            for i in node.body:
                self.visit(i)
        elif node.test.expr_ptr is False:
            for i in node.orelse:
                self.visit(i)
        else:
            with If(node.test.expr_ptr):
                self.ctx_stack.create_scope()
                for i in node.body:
                    self.visit(i)
                self.ctx_stack.pop_scope()
            if len(node.orelse) > 0:
                with Else():
                    self.ctx_stack.create_scope()
                    for i in node.orelse:
                        self.visit(i)
                    self.ctx_stack.pop_scope()

        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        for i in node.comparators:
            assert hasattr(i, "expr_ptr"), "Comparator is not an expression"
        assert hasattr(node.left, "expr_ptr"), "Comparator is not an expression"
        ops = {
            ast.Eq: lambda x, y: x == y,
            ast.NotEq: lambda x, y: x != y,
            ast.Lt: lambda x, y: x < y,
            ast.LtE: lambda x, y: x <= y,
            ast.Gt: lambda x, y: x > y,
            ast.GtE: lambda x, y: x >= y,
        }
        for i in node.ops:
            assert type(i) in ops, "Compare operator not supported"
        expr = ops[type(node.ops[0])](node.left.expr_ptr,
                                      node.comparators[0].expr_ptr)
        lf = node.comparators[0].expr_ptr
        for op, comparator in zip(node.ops[1:], node.comparators[1:]):
            expr = l_and(expr, ops[type(op)](lf, comparator.expr_ptr))
            lf = comparator.expr_ptr
        node.expr_ptr = expr
        return node

    def visit_Return(self, node):
        self.generic_visit(node)
        if self.returned:
            assert False, "The function must have no more than one return statement"
        self.returned = True
        if isinstance(node.value, ast.Name):
            name = node.value.id
            self.returns.append(self.get_name(name))
        elif isinstance(node.value, ast.Tuple):
            for value in node.value.elts:
                name = value.id
                self.returns.append(self.get_name(name))

        return node


def _get_global_vars(func):
    # From Taichi
    # Discussions: https://github.com/taichi-dev/taichi/issues/282
    import copy

    global_vars = copy.copy(func.__globals__)

    freevar_names = func.__code__.co_freevars
    closure = func.__closure__
    if closure:
        for name, value in zip(freevar_names, closure):
            try:
                global_vars[name] = value.cell_contents
            except ValueError:  # ValueError: Cell is empty
                pass

    return global_vars


def _remove_indent(lines):
    # From Taichi

    lines = lines.split("\n")
    to_remove = 0
    for i in range(len(lines[0])):
        if lines[0][i] == " ":
            to_remove = i + 1
        else:
            break

    cleaned = []
    for l in lines:
        cleaned.append(l[to_remove:])
        if len(l) >= to_remove:
            for i in range(to_remove):
                assert l[i] == " "

    return "\n".join(cleaned)


def transform(func):
    ctx_stack = ASTContextStack()
    src = _remove_indent(ins.getsource(func))
    tree = ast.parse(src)
    params = list(inspect.signature(func).parameters)
    globals = _get_global_vars(func)
    transformer = ASTTransformer(ctx_stack, params, globals)
    transformer.visit(tree)
    return Func(func.__name__, params, pop_ast(), func)


def inline(func):
    src = _remove_indent(ins.getsource(func))
    tree = ast.parse(src)
    params = list(inspect.signature(func).parameters)
    globals = _get_global_vars(func)
    funcdef = tree.body[0]
    assert isinstance(funcdef, ast.FunctionDef)
    return InlineFunction(func.__name__, funcdef.body, params, globals)
