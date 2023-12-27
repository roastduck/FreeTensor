import freetensor as ft
import pytest


def test_basic():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("t", (4,), "int32", "cache", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, t, y):
        with ft.For("i", 0, 4) as i:
            t[i] = x[i] + 1
            y[i] = t[i] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_preserve_data_type():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("t", (4,), "int32", "cache", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x, t, y):
        with ft.For("i", 0, 4) as i:
            t[i] = x[i] * 2
            y[i] = t[i] + 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = ft.cast(x[i] * 2, "int32") + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_multiple_vars():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("t", (4,), "int32", "cache", "cpu"),
                    ("u", (4,), "int32", "cache", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, t, u, y):
        with ft.For("i", 0, 4) as i:
            t[i] = x[i] + 1
            u[i] = t[i] + 1
            y[i] = u[i] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 4
    std = ft.pop_ast()

    assert std.match(ast)


def test_prop_across_loops():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("t", (4,), "int32", "cache", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, t, y):
        with ft.For("i", 0, 4) as i:
            t[i] = x[i] + 1
        with ft.For("i", 0, 4) as i:
            y[i] = t[i] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_prop_out_of_a_loop():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("t", (4,), "int32", "cache", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, t, y):
        with ft.For("i", 0, 4) as i:
            t[i] = x[i] * 2
        y[...] = t[0] + t[1] + t[2] + t[3]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[...] = x[0] * 2 + x[1] * 2 + x[2] * 2 + x[3] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_used_in_many_stmts_no_prop():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[()] = x[i] + 1
                y1[i] = t[()] * 2
                y2[i] = t[()] * 3
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[()] = x[i] + 1
                y1[i] = t[()] * 2
                y2[i] = t[()] * 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_used_in_many_reads_no_prop():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[i] = x[i] + 1
                y[i] = t[i] * t[i] + t[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[i] = x[i] + 1
                y[i] = t[i] * t[i] + t[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_used_in_many_iterations_no_prop():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[i] = x[i] + 1
                with ft.For("j", 0, 8) as j:
                    y[i, j] = t[i] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[i] = x[i] + 1
                with ft.For("j", 0, 8) as j:
                    y[i, j] = t[i] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_modify_self_no_prop():
    with ft.VarDef([("x", (5,), "float64", "inout", "cpu"),
                    ("y", (5,), "float64", "output", "cpu")]) as (x, y):
        with ft.VarDef("t", (5,), "float64", "cache", "cpu") as t:
            with ft.For("i", 0, 5) as i:
                t[i] = x[i]
                x[i] = 0
            with ft.For("i", 0, 5) as i:
                t[i] = t[i] * 2 + 1
                y[i] = t[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (5,), "float64", "inout", "cpu"),
                    ("y", (5,), "float64", "output", "cpu")]) as (x, y):
        with ft.VarDef("t", (5,), "float64", "cache", "cpu") as t:
            with ft.For("i", 0, 5) as i:
                t[i] = x[i]
                x[i] = 0
            with ft.For("i", 0, 5) as i:
                t[i] = t[i] * 2 + 1
                y[i] = t[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_using_local_var_no_prop():
    with ft.VarDef([("x", (5, 10), "float64", "input", "cpu"),
                    ("y", (5,), "float64", "output", "cpu")]) as (x, y):
        with ft.VarDef("t", (5,), "float64", "cache", "cpu") as t:
            with ft.For("i", 0, 5) as i:
                with ft.VarDef("s", (), "float64", "cache", "cpu") as s:
                    s[()] = 0
                    with ft.For("j", 0, 10) as j:
                        s[()] += x[i, j]
                    t[i] = s[()] * 2  # No prop t
            with ft.For("i", 0, 5) as i:
                y[i] = t[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (5, 10), "float64", "input", "cpu"),
                    ("y", (5,), "float64", "output", "cpu")]) as (x, y):
        with ft.VarDef("t", (5,), "float64", "cache", "cpu") as t:
            with ft.For("i", 0, 5) as i:
                with ft.VarDef("s", (), "float64", "cache", "cpu") as s:
                    s[()] = 0
                    with ft.For("j", 0, 10) as j:
                        s[()] += x[i, j]
                    t[i] = s[()] * 2  # No prop t
            with ft.For("i", 0, 5) as i:
                y[i] = t[i]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_thread_local_no_prop():
    with ft.VarDef([("x", (5, 10), "float32", "input", "gpu/global"),
                    ("y", (5, 10), "float32", "output", "gpu/global")]) as (x,
                                                                            y):
        with ft.For("i", 0, 5, label="Li") as i:
            with ft.VarDef([
                ("t", (10,), "float32", "cache", "gpu/local"),
                ("u", (10,), "float32", "cache", "gpu/shared"),
            ]) as (t, u):
                with ft.For("j", 0, 10, label="Lj1") as j:
                    t[j] = x[i, j] + (t[j - 1] if j > 0 else 0)
                    u[j] = ft.sin(t[j]) * ft.cos(t[j])
                with ft.For("j", 0, 10, label="Lj2") as j:
                    # Used `u` for only once, but we can't propagate the `t`-expression
                    # here, because `t` is thread-local
                    y[i, j] = u[j]
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.parallelize("Li", "blockIdx.x")
    s.parallelize("Lj2", "threadIdx.x")
    ast = s.ast()
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (5, 10), "float32", "input", "gpu/global"),
                    ("y", (5, 10), "float32", "output", "gpu/global")]) as (x,
                                                                            y):
        with ft.For("i", 0, 5, label="Li") as i:
            with ft.VarDef("u", (10,), "float32", "cache", "gpu/shared") as u:
                with ft.VarDef("t", (10,), "float32", "cache",
                               "gpu/local") as t:
                    with ft.For("j", 0, 10, label="Lj1") as j:
                        t[j] = x[i, j] + (t[j - 1] if j > 0 else 0)
                        u[j] = ft.sin(t[j]) * ft.cos(t[j])
                with ft.For("j", 0, 10, label="Lj2") as j:
                    y[i, j] = u[j]  # Unchanged
    std = ft.pop_ast()

    assert std.match(ast)


def test_different_iter_non_linear():
    with ft.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("y", (16,), "int32", "output", "cpu")]) as (x1, x2, y):
        ft.MarkLabel("T")
        with ft.VarDef("t", (16,), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                with ft.For("j", 0, 4) as j:
                    t[i * 4 + j] = x1[i] * x2[j]
            with ft.For("k", 0, 16) as k:
                y[k] = t[k] + 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, skip_passes=["use_builtin_div"], verbose=1)

    with ft.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("y", (16,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ft.For("k", 0, 16) as k:
            y[k] = x1[k // 4] * x2[k % 4] + 1
    std = ft.pop_ast()

    assert std.match(ast)
