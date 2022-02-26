import ir
import pytest


def test_basic():
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("p", (), "bool", "input", "cpu"),
                    ("y", (), "bool", "output", "cpu")]) as (x, p, y):
        y[()] = ir.l_and(ir.l_or(x[()] < 5, ir.l_and(x[()] >= 5, True)), p[()])
    ast = ir.pop_ast()
    print(ast)
    ast = ir.z3_simplify(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("p", (), "bool", "input", "cpu"),
                    ("y", (), "bool", "output", "cpu")]) as (x, p, y):
        y[()] = p[()]
    std = ir.pop_ast()

    assert std.match(ast)
