import freetensor as ft
import pytest


def test_basic():
    with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                y[i, j] = i * j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=2)
    merged = s.merge("L1", "L2")
    assert s.find(merged) == s.find("$merge{L1, L2}")
    ast = ft.lower(s.ast(), skip_passes=["use_builtin_div"], verbose=1)

    with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 32) as i:
            y[i // 8, i % 8] = i // 8 * (i % 8)
    std = ft.pop_ast()

    assert std.match(ast)


def test_non_zero_begin():
    with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 2, 6, label="L1") as i:
            with ft.For("j", 4, 12, label="L2") as j:
                y[i - 2, j - 4] = i * j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.merge("L1", "L2")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=["use_builtin_div"], verbose=1)

    with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 32) as i:
            y[i // 8, i % 8] = (i // 8 + 2) * ((i % 8) + 4)
    std = ft.pop_ast()

    assert std.match(ast)


def test_step():
    with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 2, -2, -2, label="L1") as i:
            with ft.For("j", 6, -2, -2, label="L2") as j:
                y[i, j] = i * j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.merge("L1", "L2")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=["use_builtin_div"], verbose=1)

    with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8) as i:
            y[2 + i // 4 * -2,
              6 + i % 4 * -2] = (2 + i // 4 * -2) * (6 + i % 4 * -2)
    std = ft.pop_ast()

    assert std.match(ast)


def test_invalid():
    with ft.VarDef("y", (4, 4, 4), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 4, label="L2") as j:
                with ft.For("k", 0, 4, label="L3") as k:
                    y[i, j, k] = i + j + k
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.merge("L1", "L3")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_if_in_between():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.If(x[i] > 0):
                with ft.For("j", 0, 8, label="L2") as j:
                    y[i, j] = i * j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.merge("L1", "L2")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=["use_builtin_div"], verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 32) as i:
            with ft.If(x[i // 8] > 0):
                y[i // 8, i % 8] = i // 8 * (i % 8)
    std = ft.pop_ast()

    assert std.match(ast)


def test_stmt_in_between():
    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4,), "int32", "output", "cpu")]) as (y, z):
        with ft.For("i", 0, 4, label="L1") as i:
            z[i] = i
            with ft.For("j", 0, 8, label="L2") as j:
                y[i, j] = i * j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.merge("L1", "L2")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=["use_builtin_div"], verbose=1)

    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4,), "int32", "output", "cpu")]) as (y, z):
        with ft.For("i", 0, 32) as i:
            with ft.If(i % 8 == 0):
                z[i // 8] = i // 8
            y[i // 8, i % 8] = i // 8 * (i % 8)
    std = ft.pop_ast()

    assert std.match(ast)


def test_loop_in_between():
    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4, 8), "int32", "output", "cpu")]) as (y, z):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                z[i, j] = i + j
            with ft.For("j", 0, 8, label="L3") as j:
                y[i, j] = i * j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.merge("L1", "L3")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=["use_builtin_div"], verbose=1)

    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4, 8), "int32", "output", "cpu")]) as (y, z):
        with ft.For("i", 0, 32) as i:
            with ft.If(i % 8 == 0):
                with ft.For("j", 0, 8) as j:
                    z[i // 8, j] = i // 8 + j
            y[i // 8, i % 8] = i // 8 * (i % 8)
    std = ft.pop_ast()

    assert std.match(ast)


def test_def_in_between():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.VarDef("z", (), "int32", "output", "cpu") as z:
                z[()] = x[i]
                with ft.For("j", 0, 8, label="L2") as j:
                    y[i, j] = z[()] * j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.merge("L1", "L2")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=["use_builtin_div"], verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (), "int32", "output", "cpu")]) as (x, y, z):
        with ft.For("i", 0, 32) as i:
            with ft.If(i % 8 == 0):
                z[()] = x[i // 8]
            y[i // 8, i % 8] = z[()] * (i % 8)
    std = ft.pop_ast()

    assert std.match(ast)


def test_prop_expr_of_outer_loop():
    with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.VarDef("z", (), "int32", "output", "cpu") as z:
                z[()] = i
                with ft.For("j", 0, 8, label="L2") as j:
                    y[i, j] = z[()] * j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.merge("L1", "L2")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=["use_builtin_div"], verbose=1)

    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (), "int32", "output", "cpu")]) as (y, z):
        with ft.For("i", 0, 32) as i:
            with ft.If(i % 8 == 0):
                z[()] = i // 8
            y[i // 8, i % 8] = i // 8 * (i % 8)
    std = ft.pop_ast()

    assert std.match(ast)


def test_no_merge_if_outer_iter_var_is_used_in_inner():
    with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, i, label="L2") as j:
                y[i, j] = i * j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=2)
    with pytest.raises(ft.InvalidSchedule):
        s.merge("L1", "L2")
