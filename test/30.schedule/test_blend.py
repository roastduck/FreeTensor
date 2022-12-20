import freetensor as ft
import pytest


def test_basic():
    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4, label="L1") as i:
            y1[i] = i + 1
            y2[i] = i + 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        y1[0] = 1
        y1[1] = 2
        y1[2] = 3
        y1[3] = 4
        y2[0] = 2
        y2[1] = 3
        y2[2] = 4
        y2[3] = 5
    std = ft.pop_ast()

    assert std.match(ast)


def test_begin_and_step():
    with ft.VarDef([("y1", (8,), "int32", "output", "cpu"),
                    ("y2", (8,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 6, -2, -2, label="L1") as i:
            y1[i] = i + 1
            y2[i] = i + 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("y1", (8,), "int32", "output", "cpu"),
                    ("y2", (8,), "int32", "output", "cpu")]) as (y1, y2):
        y1[6] = 7
        y1[4] = 5
        y1[2] = 3
        y1[0] = 1
        y2[6] = 8
        y2[4] = 6
        y2[2] = 4
        y2[0] = 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_inner_if():
    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.If(x[i] > 0):
                y1[i] = i + 1
                y2[i] = i + 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.If(x[0] > 0):
            y1[0] = 1
        with ft.If(x[1] > 0):
            y1[1] = 2
        with ft.If(x[2] > 0):
            y1[2] = 3
        with ft.If(x[3] > 0):
            y1[3] = 4
        with ft.If(x[0] > 0):
            y2[0] = 2
        with ft.If(x[1] > 0):
            y2[1] = 3
        with ft.If(x[2] > 0):
            y2[2] = 4
        with ft.If(x[3] > 0):
            y2[3] = 5
    std = ft.pop_ast()

    assert std.match(ast)


def test_inner_if_fuse():
    with ft.VarDef([
        ("x", (), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.If(x[()] > 0):
                y1[i] = i + 1
                y2[i] = i + 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("x", (()), "int32", "input", "cpu") as x:
        with ft.If(x[()] > 0):
            with ft.VarDef([
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
    std = ft.pop_ast()

    assert std.match(ast)


def test_inner_if_else():
    with ft.VarDef([
        ("x", (2,), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "output", "cpu"),
        ("y2", (2,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 2, label="L1") as i:
            with ft.If(x[i] > 0):
                y1[i] = i + 1
                y2[i] = i + 2
            with ft.Else():
                y1[i] = i - 1
                y2[i] = i - 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("x", (2,), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "output", "cpu"),
        ("y2", (2,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.If(x[0] > 0):
            y1[0] = 1
        with ft.If(x[1] > 0):
            y1[1] = 2
        with ft.If(x[0] > 0):
            y2[0] = 2
        with ft.If(x[1] > 0):
            y2[1] = 3
        with ft.If(x[0] <= 0):
            y1[0] = -1
        with ft.If(x[1] <= 0):
            y1[1] = 0
        with ft.If(x[0] <= 0):
            y2[0] = -2
        with ft.If(x[1] <= 0):
            y2[1] = -1
    std = ft.pop_ast()

    assert std.match(ast)


def test_inner_for():
    with ft.VarDef([
        ("x", (2,), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "inout", "cpu"),
        ("y2", (2,), "int32", "inout", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 2, label="L1") as i:
            with ft.For("j", 0, x[i]):
                y1[i] *= 2
                y2[i] *= 3
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("x", (2,), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "inout", "cpu"),
        ("y2", (2,), "int32", "inout", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("j", 0, x[0]):
            y1[0] *= 2
        with ft.For("j", 0, x[1]):
            y1[1] *= 2
        with ft.For("j", 0, x[0]):
            y2[0] *= 3
        with ft.For("j", 0, x[1]):
            y2[1] *= 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_inner_for_fuse():
    with ft.VarDef([
        ("x", (), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "inout", "cpu"),
        ("y2", (2,), "int32", "inout", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 2, label="L1") as i:
            with ft.For("j", 0, x[()]):
                y1[i] *= 2
                y2[i] *= 3
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("x", (()), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "inout", "cpu"),
        ("y2", (2,), "int32", "inout", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("j", 0, x[()]):
            y1[0] *= 2
            y1[1] *= 2
            y2[0] *= 3
            y2[1] *= 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_inner_for_fuse_different_begin():
    with ft.VarDef([
        ("x", (), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "inout", "cpu"),
        ("y2", (2,), "int32", "inout", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 2, label="L1") as i:
            with ft.For("j", i, x[()] + i):
                y1[i] *= 2
                y2[i] *= 3
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("x", (()), "int32", "input", "cpu"),
        ("y1", (2,), "int32", "inout", "cpu"),
        ("y2", (2,), "int32", "inout", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("j", 0, x[()]):
            y1[0] *= 2
            y1[1] *= 2
            y2[0] *= 3
            y2[1] *= 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_unsolvable_dependence():
    with ft.VarDef([("y1", (), "int32", "inout", "cpu"),
                    ("y2", (), "int32", "inout", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 2, label="L1") as i:
            y1[()] = y2[()] * i + 1
            y2[()] = y2[()] * i + 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.blend("L1")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_loop_not_found():
    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4, label="L1") as i:
            y1[i] = i + 1
            y2[i] = i + 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.blend("L2")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_var_def_inside():
    with ft.VarDef([("x", (2,), "int32", "input", "cpu"),
                    ("y1", (2,), "int32", "output", "cpu"),
                    ("y2", (2,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 2, label="L1") as i:
            with ft.VarDef("b", (), "int32", "cache", "cpu") as b:
                b[()] = x[i] * 2
                y1[i] = b[()] + 1
                y2[i] = b[()] + 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (2,), "int32", "input", "cpu"),
                    ("y1", (2,), "int32", "output", "cpu"),
                    ("y2", (2,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("b.0", (), "int32", "cache", "cpu") as b0:
            b0[()] = x[0] * 2
            with ft.VarDef("b.1", (), "int32", "cache", "cpu") as b1:
                b1[()] = x[1] * 2
                y1[0] = b0[()] + 1
                y1[1] = b1[()] + 1
                y2[0] = b0[()] + 2
                y2[1] = b1[()] + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_var_def_inside_no_need_to_split():
    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("x", (2,), "int32", "input", "cpu"),
                    ("y", (2,), "int32", "output", "cpu")]) as (n, x, y):
        with ft.For("i", 0, 2, label="L1") as i:
            with ft.VarDef("b", (), "int32", "cache", "cpu") as b:
                b[()] = n[()]
                y[i] = b[()] * x[i]
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.blend("L1")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("x", (2,), "int32", "input", "cpu"),
                    ("y", (2,), "int32", "output", "cpu")]) as (n, x, y):
        with ft.VarDef("b", (), "int32", "cache", "cpu") as b:
            b[()] = n[()]
            y[0] = b[()] * x[0]
            y[1] = b[()] * x[1]
    std = ft.pop_ast()

    assert std.match(ast)
