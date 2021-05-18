import ast
import numpy as np
import sourceinspect as ins
from typing import Sequence

from .nodes import _VarDef, Var, pop_ast, For, If, Else, MarkNid, ctx_stack as node_ctx
from .utils import *
import ffi
import sys

assert sys.version_info >= (3,
                            8), "Python version lower than 3.8 is not supported"


def declare_var(var, shape, dtype, atype, mtype):
    pass


def create_var(shape, dtype, atype, mtype):
    return np.zeros(shape, dtype)


class VarSubscript:

    def __init__(self, var, sub):
        self.var = var
        self.sub = sub

    def set(self, value):
        self.var.__setitem__(self.sub, value)


class VarCreation:

    def __init__(self, shape: Sequence, dtype, atype, mtype, name=None):
        self.shape = shape
        self.dtype = dtype
        self.atype = atype
        self.mtype = mtype
        self.name = name

    def add_name(self, name):
        assert self.name is None, "Bug: Variable name is set more than once"
        self.name = name

    def execute(self):
        assert self.name is not None, "Bug: Variable name is not set"
        ctx_stack.create_variable(self.name, self.shape, self.dtype, self.atype,
                                  self.mtype)


class ASTContext:

    def __init__(self):
        self.vardef_stack = []
        self.var_dict = {}
        self.old_vars = []


class ASTContextStack:

    def __init__(self):
        self.ctx_stack = []
        self.now_var_id = {}
        self.name_set = set()
        self.next_nid = ""

    def clear(self):
        self.ctx_stack = []
        self.now_var_id = {}
        self.name_set = set()
        self.next_nid = ""

    def top(self) -> ASTContext:
        return self.ctx_stack[-1]

    def get_current_name(self, name, prefetch=False):
        name_id = self.now_var_id.get(name)
        if prefetch and name_id is None:
            return None
        assert name_id is not None, "Variable %s not found" % name
        if name_id != 0:
            return "___cache_" + name + "_" + str(name_id)
        return name

    def create_current_name(self, name, atype):
        if atype != "cache":
            if name in self.name_set:
                assert False, "Non-cache variables cannot be redefined"
            self.name_set.add(name)
            self.now_var_id[name] = 0
            return name
        name_id = self.now_var_id.get(name)
        if name_id is None:
            name_id = 1
        else:
            name_id += 1
        while "___cache_" + name + "_" + str(name_id) in self.name_set:
            name_id += 1
        self.now_var_id[name] = name_id
        if name_id:
            name = "___cache_" + name + "_" + str(name_id)
        self.name_set.add(name)
        return name

    def find_var_by_name(self, name, prefetch=False):
        name = self.get_current_name(name, prefetch)

        for ctx in reversed(self.ctx_stack):  # type: ASTContext
            if name in ctx.old_vars:
                if prefetch:
                    return None
                assert False, "Variable reassigned in if/for/while"
            var = ctx.var_dict.get(name)
            if var is not None:
                return var

        if prefetch:
            return None
        assert False, "Bug: variable not found by find_var_by_name"

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

    def create_variable(self, name, shape, dtype, atype, mtype):
        name = self.create_current_name(name, atype)
        vardef = _VarDef(name, shape, dtype, atype, mtype)
        var = vardef.__enter__()
        top = self.top()
        top.vardef_stack.append(vardef)
        top.var_dict[name] = var
        return var

    def create_loop(self, name, begin, end):
        name = self.create_current_name(name, "cache")
        fr = For(name, begin, end, self.get_nid())
        var = fr.__enter__()
        top = self.top()
        top.var_dict[name] = var
        return fr

    def set_nid(self, name):
        MarkNid(name)

    def get_nid(self):
        ret = node_ctx.top().get_next_nid()
        MarkNid("")
        return ret


ctx_stack = ASTContextStack()


class ASTTransformer(ast.NodeTransformer):

    def __init__(self):
        super().__init__()

    @staticmethod
    def parse_stmt(stmt):
        return ast.parse(stmt).body[0]

    @staticmethod
    def parse_expr(expr):
        return ast.parse(expr).body[0].value

    def visit_Name(self, node):
        node.expr_ptr = ctx_stack.find_var_by_name(node.id, prefetch=True)
        return node

    def visit_Constant(self, node):
        value = node.value
        if (isinstance(value, int) or isinstance(value, float) or
                isinstance(value, bool)):
            node.expr_ptr = value
        elif isinstance(value, str):
            node.expr_str = value
        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if node.value.expr_ptr is None:
            var = ctx_stack.find_var_by_name(node.value.id)
        assert isinstance(node.value.expr_ptr, Var), "Invalid subscript"
        var = node.value.expr_ptr
        value = node.slice.value
        if isinstance(value, ast.Tuple):
            assert hasattr(value, "expr_tuple")
            tup = value.expr_tuple
        elif hasattr(value, "expr_ptr"):
            tup = (value.expr_ptr,)
        else:
            assert False, "Invalid subscript value"
        if isinstance(node.ctx, ast.Load):
            node.expr_ptr = var[tup]
        else:
            node.expr_ptr = VarSubscript(var, tup)
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
            ast.And: lambda l, r: ffi.makeLAnd(l, r),
            ast.Or: lambda l, r: ffi.makeLOr(l, r),
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
            ast.Not: lambda l: ffi.makeLNot(l),
        }.get(type(node.op))
        assert op is not None, "Unary operator not implemented"
        node.expr_ptr = op(node.operand.expr_ptr)
        return node

    def visit_FunctionDef(self, node):
        ctx_stack.create_scope()
        self.generic_visit(node)
        ctx_stack.pop_scope()
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == "ir"):
            func_name = node.func.attr
            args = node.args
            if func_name == "create_var":
                assert len(args) == 4, "create_var function has 4 arguments"
                shape = args[0]
                assert hasattr(shape, "expr_tuple"), "Shape is not a tuple"
                shape = shape.expr_tuple
                for i in range(1, 4):
                    assert hasattr(
                        args[i],
                        "expr_str"), "argument {} expects str".format(i + 1)
                dtype = args[1].expr_str
                atype = args[2].expr_str
                mtype = args[3].expr_str
                node.expr_call = VarCreation(shape, dtype, atype, mtype)
            elif func_name == "declare_var":
                assert len(args) == 5, "define_var function has 5 arguments"
                assert isinstance(args[0], ast.Name)
                name = args[0].id
                shape = args[1]
                assert hasattr(shape, "expr_tuple"), "Shape is not a tuple"
                shape = shape.expr_tuple
                for i in range(2, 5):
                    assert hasattr(
                        args[i],
                        "expr_str"), "argument {} expects str".format(i + 1)
                dtype = args[2].expr_str
                atype = args[3].expr_str
                mtype = args[4].expr_str
                node.expr_call = VarCreation(shape, dtype, atype, mtype, name)
            elif func_name == "MarkNid":
                assert len(args) == 1, "MarkNid has one argument"
                assert hasattr(args[0], "expr_str"), "Nid should be string"
                nid = args[0].expr_str
                ctx_stack.set_nid(nid)
            elif func_name == "intrinsic":
                assert len(args) >= 1, "intrinsic has at least one argument"
                assert hasattr(
                    args[0], "expr_str"
                ), "The first argument of intrinsic should be string"
                expr_args = []
                for i in args[1:]:
                    assert hasattr(
                        i, "expr_ptr"), "intrinsic argument is not expression"
                    expr_args.append(i.expr_ptr)
                ret_type = ffi.DataType.Void
                for item in node.keywords:
                    assert item.arg == "ret_type", (
                        "Unrecognized keyword argument %s" % item.arg)
                    ret_type = parseDType(item.value.expr_str)
                node.expr_ptr = ffi.makeIntrinsic(args[0].expr_str,
                                                  tuple(expr_args), ret_type)
            else:
                assert False, "Function %s not implemented" % func_name
        else:
            assert False, "External call not implemented"
        return node

    def visit_Tuple(self, node):
        self.generic_visit(node)
        tup = []
        for i in node.elts:  # TODO: maybe support for tuple in tuple
            assert hasattr(i, "expr_ptr"), "Invalid tuple"
            tup.append(i.expr_ptr)
        tup = tuple(tup)
        node.expr_tuple = tup
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        print(ast.dump(node))
        # TODO: (maybe) support for multiple assignment
        assert len(node.targets) == 1, "Multiple assignment is not supported"
        if hasattr(node.value, "expr_call") and isinstance(
                node.value.expr_call, VarCreation):
            name = node.targets[0].id
            var_creation = node.value.expr_call
            var_creation.add_name(name)
            var_creation.execute()
        elif hasattr(node.targets[0], "expr_ptr") and isinstance(
                node.targets[0].expr_ptr, VarSubscript):
            var = node.targets[0].expr_ptr
            if var is None:
                var = ctx_stack.find_var_by_name(node.targets[0].id)
            assert hasattr(
                node.value,
                "expr_ptr"), "Value to be assigned is not an expression"
            var.set(node.value.expr_ptr)
        else:
            assert False, "Invalid assignment"
        return node

    def visit_For(self, node):
        if (isinstance(node.iter, ast.Call) and
                isinstance(node.iter.func, ast.Name) and
                node.iter.func.id == "range" and len(node.iter.args) == 2):

            ctx_stack.create_scope()
            name = node.target.id
            self.visit(node.iter.args[0])
            self.visit(node.iter.args[1])
            assert hasattr(node.iter.args[0],
                           "expr_ptr"), "For range is not expression"
            assert hasattr(node.iter.args[1],
                           "expr_ptr"), "For range is not expression"
            begin = node.iter.args[0].expr_ptr
            end = node.iter.args[1].expr_ptr
            fr = ctx_stack.create_loop(name, begin, end)
            for i in node.body:
                self.visit(i)
            ctx_stack.pop_scope()
            fr.__exit__(None, None, None)
        else:
            assert False, "For statement other than range(a, b) is not implemented"
        return node

    def visit_Expr(self, node):
        self.generic_visit(node)
        print(ast.dump(node))
        if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str):
            s = node.value.value
            if s[0:5] == "nid: ":
                ctx_stack.set_nid(s[5:])
        elif hasattr(node.value, "expr_call") and isinstance(
                node.value.expr_call, VarCreation):
            node.value.expr_call.execute()
        return node

    def visit_If(self, node):
        self.visit(node.test)
        assert hasattr(node.test,
                       "expr_ptr"), "If condition is not an expression"
        with If(node.test.expr_ptr):
            ctx_stack.create_scope()
            for i in node.body:
                self.visit(i)
            ctx_stack.pop_scope()
        with Else():
            ctx_stack.create_scope()
            for i in node.orelse:
                self.visit(i)
            ctx_stack.pop_scope()
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
            expr = ffi.makeLAnd(expr, ops[type(op)](lf, comparator.expr_ptr))
            lf = comparator.expr_ptr
        node.expr_ptr = expr
        return node


def _get_global_vars(func):
    # Discussions: https://github.com/taichi-dev/taichi/issues/282
    import copy

    global_vars = copy.copy(func.__globals__)

    freevar_names = func.__code__.co_freevars
    closure = func.__closure__
    if closure:
        freevar_values = list(map(lambda x: x.cell_contents, closure))
        for name, value in zip(freevar_names, freevar_values):
            global_vars[name] = value

    return global_vars


def remove_indent(lines):
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
    ctx_stack.clear()
    src = remove_indent(ins.getsource(func))
    tree = ast.parse(src)
    ASTTransformer().visit(tree)
    print(ast.dump(tree))
    return pop_ast()
