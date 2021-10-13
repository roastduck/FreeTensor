import ir
import pytest


def test_basic():
    with ir.VarDef([("a", (1000,), "int32", "input", "cpu"),
                    ("b", (1000,), "int32", "input", "cpu"),
                    ("c", (1000,), "int32", "input", "cpu"),
                    ("y", (1000,), "int32", "output", "cpu")]) as (a, b, c, y):
        with ir.For("i", 0, 1000, nid="L1") as i:
            y[i] = a[i] + b[i]
        with ir.For("i", 0, 1000, nid="L2") as i:
            y[i] += c[i]

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.auto_fuse(ir.CPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == ["fuse(L1, L2)"]


def test_nested():
    with ir.VarDef([("a", (10, 10), "int32", "input", "cpu"),
                    ("b", (10, 10), "int32", "input", "cpu"),
                    ("c", (10, 10), "int32", "input", "cpu"),
                    ("y", (10, 10), "int32", "output", "cpu")]) as (a, b, c, y):
        with ir.For("i", 0, 10, nid="L1") as i:
            with ir.For("j", 0, 10, nid="L2") as j:
                y[i, j] = a[i, j] + b[i, j]
        with ir.For("i", 0, 10, nid="L3") as i:
            with ir.For("j", 0, 10, nid="L4") as j:
                y[i, j] += c[i, j]

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.auto_fuse(ir.CPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == ["fuse(L1, L3)", "fuse(L2, L4)"]
