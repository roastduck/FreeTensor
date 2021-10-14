import ir
import pytest


def test_basic():
    with ir.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                y[i, j] = i + j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.reorder(["L2", "L1"])
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ir.For("j", 0, 8) as j:
            with ir.For("i", 0, 4) as i:
                y[i, j] = i + j
    std = ir.pop_ast()

    assert std.match(ast)


def test_multiple_loops():
    with ir.VarDef("y", (4, 8, 16), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                with ir.For("k", 0, 16, nid="L3") as k:
                    y[i, j, k] = (i + j) * k
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.reorder(["L3", "L2", "L1"])
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (4, 8, 16), "int32", "output", "cpu") as y:
        with ir.For("k", 0, 16) as k:
            with ir.For("j", 0, 8) as j:
                with ir.For("i", 0, 4) as i:
                    y[i, j, k] = (i + j) * k
    std = ir.pop_ast()

    assert std.match(ast)


def test_if_in_between():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.If(x[i] > 0):
                with ir.For("j", 0, 8, nid="L2") as j:
                    y[i, j] = i + j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.reorder(["L2", "L1"])
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        with ir.For("j", 0, 8) as j:
            with ir.For("i", 0, 4) as i:
                with ir.If(x[i] > 0):
                    y[i, j] = i + j
    std = ir.pop_ast()

    assert std.match(ast)


def test_stmt_in_between():
    with ir.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4,), "int32", "output", "cpu")]) as (y, z):
        with ir.For("i", 0, 4, nid="L1") as i:
            z[i] = i
            with ir.For("j", 0, 8, nid="L2") as j:
                y[i, j] = i + j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.reorder(["L2", "L1"])
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4,), "int32", "output", "cpu")]) as (y, z):
        with ir.For("j", 0, 8) as j:
            with ir.For("i", 0, 4) as i:
                with ir.If(j == 0):
                    z[i] = i
                y[i, j] = i + j
    std = ir.pop_ast()

    assert std.match(ast)


def test_dependency():
    with ir.VarDef("y", (1,), "int32", "output", "cpu") as y:
        y[0] = 0
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                y[0] = y[0] * i + j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.reorder(["L2", "L1"])
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_reduction():
    with ir.VarDef([("x", (4, 8), "int32", "output", "cpu"),
                    ("y", (1,), "int32", "output", "cpu")]) as (x, y):
        y[0] = 0
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                y[0] = y[0] + x[i, j]
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.reorder(["L2", "L1"])
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4, 8), "int32", "output", "cpu"),
                    ("y", (1,), "int32", "output", "cpu")]) as (x, y):
        y[0] = 0
        with ir.For("j", 0, 8) as j:
            with ir.For("i", 0, 4) as i:
                ir.Any()
    std = ir.pop_ast()

    assert std.match(ast)


def test_local_var():
    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y1", (4, 8), "int32", "output", "cpu"),
        ("y2", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y1, y2):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                with ir.VarDef("buf", (1,), "int32", "cache", "cpu") as buf:
                    buf[0] = x0[i, j] + x1[i, j]
                    y1[i, j] = buf[0] * 2
                    y2[i, j] = buf[0] * 3
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.reorder(["L2", "L1"])
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y1", (4, 8), "int32", "output", "cpu"),
        ("y2", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y1, y2):
        with ir.For("j", 0, 8) as j:
            with ir.For("i", 0, 4) as i:
                with ir.VarDef("buf", (1,), "int32", "cache", "cpu") as buf:
                    buf[0] = x0[i, j] + x1[i, j]
                    y1[i, j] = buf[0] * 2
                    y2[i, j] = buf[0] * 3
    std = ir.pop_ast()

    assert std.match(ast)
