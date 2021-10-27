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


def test_stmt_in_between_1():
    with ir.VarDef([("x1", (1000,), "int32", "input", "cpu"),
                    ("x2", (1000,), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x1, x2, y1, y2):
        ir.MarkNid('S1')
        y1[()] = 0
        with ir.For("i", 0, 1000, nid="L1") as i:
            y1[()] += x1[i]
        ir.MarkNid('S2')
        y2[()] = 0
        with ir.For("i", 0, 1000, nid="L2") as i:
            y2[()] += x2[i]

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.auto_fuse(ir.CPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == ["swap(S2, L1)", "fuse(L1, L2)"]


def test_stmt_in_between_2():
    with ir.VarDef([("x1", (), "int32", "inout", "cpu"),
                    ("x2", (), "int32", "inout", "cpu"),
                    ("y1", (1000,), "int32", "output", "cpu"),
                    ("y2", (1000,), "int32", "output", "cpu")]) as (x1, x2, y1,
                                                                    y2):
        with ir.For("i", 0, 1000, nid="L1") as i:
            y1[i] = x1[()] * i
        ir.MarkNid('S1')
        x1[()] = 0
        with ir.For("i", 0, 1000, nid="L2") as i:
            y2[i] = x2[()] * i
        ir.MarkNid('S2')
        x2[()] = 0

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.auto_fuse(ir.CPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == ["swap(L2, S1)", "fuse(L1, L2)"]
