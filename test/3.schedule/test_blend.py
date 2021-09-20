import ir
import pytest


def test_basic():
    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4, nid="L1") as i:
            y1[i] = i + 1
            y2[i] = i + 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        y1[0] = 1
        y1[1] = 2
        y1[2] = 3
        y1[3] = 4
        y2[0] = 2
        y2[1] = 3
        y2[2] = 4
        y2[3] = 5
    std = ir.pop_ast()

    assert std.match(ast)


def test_inner_if():
    with ir.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.If(x[i] > 0):
                y1[i] = i + 1
                y2[i] = i + 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ir.If(x[0] > 0):
            y1[0] = 1
        with ir.If(x[1] > 0):
            y1[1] = 2
        with ir.If(x[2] > 0):
            y1[2] = 3
        with ir.If(x[3] > 0):
            y1[3] = 4
        with ir.If(x[0] > 0):
            y2[0] = 2
        with ir.If(x[1] > 0):
            y2[1] = 3
        with ir.If(x[2] > 0):
            y2[2] = 4
        with ir.If(x[3] > 0):
            y2[3] = 5
    std = ir.pop_ast()

    assert std.match(ast)


def test_inner_if_fuse():
    with ir.VarDef([
        ("x", (), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.If(x[()] > 0):
                y1[i] = i + 1
                y2[i] = i + 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("x", (()), "int32", "input", "cpu") as x:
        with ir.If(x[()] > 0):
            with ir.VarDef([
                ("y1", (4,), "int32", "output", "cpu"),
                ("y2", (4,), "int32", "output", "cpu"),
            ]) as (y1, y2):
                y1[0] = 1
                y1[1] = 2
                y1[2] = 3
                y1[3] = 4
                y2[0] = 2
                y2[1] = 3
                y2[2] = 4
                y2[3] = 5
    std = ir.pop_ast()

    assert std.match(ast)


def test_inner_if_else():
    with ir.VarDef([
        ("x", (2,), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "output", "cpu"),
        ("y2", (2,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ir.For("i", 0, 2, nid="L1") as i:
            with ir.If(x[i] > 0):
                y1[i] = i + 1
                y2[i] = i + 2
            with ir.Else():
                y1[i] = i - 1
                y2[i] = i - 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x", (2,), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "output", "cpu"),
        ("y2", (2,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ir.If(x[0] > 0):
            y1[0] = 1
        with ir.If(x[1] > 0):
            y1[1] = 2
        with ir.If(x[0] > 0):
            y2[0] = 2
        with ir.If(x[1] > 0):
            y2[1] = 3
        with ir.If(x[0] <= 0):
            y1[0] = -1
        with ir.If(x[1] <= 0):
            y1[1] = 0
        with ir.If(x[0] <= 0):
            y2[0] = -2
        with ir.If(x[1] <= 0):
            y2[1] = -1
    std = ir.pop_ast()

    assert std.match(ast)


def test_inner_for():
    with ir.VarDef([
        ("x", (2,), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "inout", "cpu"),
        ("y2", (2,), "int32", "inout", "cpu"),
    ]) as (x, y1, y2):
        with ir.For("i", 0, 2, nid="L1") as i:
            with ir.For("j", 0, x[i]):
                y1[i] = y1[i] * 2
                y2[i] = y2[i] * 3
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x", (2,), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "inout", "cpu"),
        ("y2", (2,), "int32", "inout", "cpu"),
    ]) as (x, y1, y2):
        with ir.For("j", 0, x[0]):
            y1[0] = y1[0] * 2
        with ir.For("j", 0, x[1]):
            y1[1] = y1[1] * 2
        with ir.For("j", 0, x[0]):
            y2[0] = y2[0] * 3
        with ir.For("j", 0, x[1]):
            y2[1] = y2[1] * 3
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_inner_for_fuse():
    with ir.VarDef([
        ("x", (), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "inout", "cpu"),
        ("y2", (2,), "int32", "inout", "cpu"),
    ]) as (x, y1, y2):
        with ir.For("i", 0, 2, nid="L1") as i:
            with ir.For("j", 0, x[()]):
                y1[i] = y1[i] * 2
                y2[i] = y2[i] * 3
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x", (()), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "inout", "cpu"),
        ("y2", (2,), "int32", "inout", "cpu"),
    ]) as (x, y1, y2):
        with ir.For("j", 0, x[()]):
            y1[0] = y1[0] * 2
            y1[1] = y1[1] * 2
            y2[0] = y2[0] * 3
            y2[1] = y2[1] * 3
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_inner_for_fuse_different_begin():
    with ir.VarDef([
        ("x", (), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "inout", "cpu"),
        ("y2", (2,), "int32", "inout", "cpu"),
    ]) as (x, y1, y2):
        with ir.For("i", 0, 2, nid="L1") as i:
            with ir.For("j", i, x[()] + i):
                y1[i] = y1[i] * 2
                y2[i] = y2[i] * 3
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x", (()), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "inout", "cpu"),
        ("y2", (2,), "int32", "inout", "cpu"),
    ]) as (x, y1, y2):
        with ir.For("j", 0, x[()]):
            y1[0] = y1[0] * 2
            y1[1] = y1[1] * 2
            y2[0] = y2[0] * 3
            y2[1] = y2[1] * 3
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_unsolvable_dependency():
    with ir.VarDef([("y1", (), "int32", "inout", "cpu"),
                    ("y2", (), "int32", "inout", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 2, nid="L1") as i:
            y1[()] = y2[()] * i
            y2[()] = y2[()] + i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.blend("L1")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_loop_not_found():
    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4, nid="L1") as i:
            y1[i] = i + 1
            y2[i] = i + 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.blend("L2")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_var_def_inside():
    with ir.VarDef([("x", (2,), "int32", "input", "cpu"),
                    ("y", (2,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 2, nid="L1") as i:
            with ir.VarDef("b", (), "int32", "cache", "cpu") as b:
                b[()] = x[i] * 2
                y[i] = b[()] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (2,), "int32", "input", "cpu"),
                    ("y", (2,), "int32", "output", "cpu")]) as (x, y):
        with ir.VarDef("b.0", (), "int32", "cache", "cpu") as b0:
            b0[()] = x[0] * 2
            with ir.VarDef("b.1", (), "int32", "cache", "cpu") as b1:
                b1[()] = x[1] * 2
                y[0] = b0[()] + 1
                y[1] = b1[()] + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_var_def_inside_no_need_to_split():
    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("x", (2,), "int32", "input", "cpu"),
                    ("y", (2,), "int32", "output", "cpu")]) as (n, x, y):
        with ir.For("i", 0, 2, nid="L1") as i:
            with ir.VarDef("b", (), "int32", "cache", "cpu") as b:
                b[()] = n[()]
                y[i] = b[()] * x[i]
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("x", (2,), "int32", "input", "cpu"),
                    ("y", (2,), "int32", "output", "cpu")]) as (n, x, y):
        with ir.VarDef("b", (), "int32", "cache", "cpu") as b:
            b[()] = n[()]
            y[0] = b[()] * x[0]
            y[1] = b[()] * x[1]
    std = ir.pop_ast()

    assert std.match(ast)
