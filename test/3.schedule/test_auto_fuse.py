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
