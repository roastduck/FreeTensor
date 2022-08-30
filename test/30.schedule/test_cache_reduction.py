import freetensor as ft
import pytest


def test_reduce_sum():
    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                # Intended to left `y = y + ...` to ensure it invokes make_reduction
                y[i, j] = y[i, j] + x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.cache_reduction("L2", "y", "cpu")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=['prop_one_time_use'], verbose=1)

    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("b", (1, 8), "int32", "cache", "cpu") as b:
                with ft.For("j", 0, 8) as j:
                    b[0, j] = 2 * x[i, j]  # After remove_writes pass
                with ft.For("j1", 0, 8) as j:
                    y[i, j] += b[0, j]
    std = ft.pop_ast()

    assert std.match(ast)


def test_reduce_sum_loop():
    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                # Intended to left `y = y + ...` to ensure it invokes make_reduction
                y[i] = y[i] + x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.cache_reduction("L2", "y", "cpu")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("b", (1,), "int32", "cache", "cpu") as b:
                b[0] = 0
                with ft.For("j2", 0, 8) as j:
                    b[0] += x[i, j] * 2
                y[i] += b[0]
    std = ft.pop_ast()

    assert std.match(ast)


def test_reduce_min_loop():
    with ft.VarDef([
        ("x", (4, 8), "float32", "input", "cpu"),
        ("y", (4,), "float32", "inout", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                y[i] = ft.min(y[i], x[i, j] * 2)
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.cache_reduction("L2", "y", "cpu")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("x", (4, 8), "float32", "input", "cpu"),
        ("y", (4,), "float32", "inout", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("b", (1,), "float32", "cache", "cpu") as b:
                b[0] = float("inf")
                with ft.For("j2", 0, 8) as j:
                    b[0] = ft.min(b[0], x[i, j] * 2)
                y[i] = ft.min(y[i], b[0])
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_no_var():
    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                ft.MarkLabel("S0")
                y[i, j] = y[i, j] + x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.cache_reduction("S0", "z", "cpu")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_no_stmt():
    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                y[i, j] = y[i, j] + x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.cache_reduction("S0", "x", "cpu")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_read_not_allowed():
    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            y[i] = 0
            with ft.For("j", 0, 8, label="L2") as j:
                ft.MarkLabel("S0")
                y[i] = y[i] + x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.cache_reduction("S0", "x", "cpu")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_write_not_allowed():
    with ft.VarDef([
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                ft.MarkLabel("S0")
                y[i, j] = x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.cache_reduction("S0", "y", "cpu")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_mix_op_not_allowed():
    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                with ft.NamedScope("S0"):
                    y[i, j] = y[i, j] + x[i, j] * 2
                    y[i, j] = ft.min(y[i, j], x[i, j] * 2)
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.cache_reduction("S0", "y", "cpu")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)
