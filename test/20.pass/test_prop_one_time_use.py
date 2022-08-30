import freetensor as ft


def test_basic():
    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("t", (4,), "int32", "cache", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, t, y):
        with ft.For("i", 0, 4) as i:
            t[i] = x[i] + 1
            y[i] = t[i] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_multiple_vars():
    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("t", (4,), "int32", "cache", "cpu"),
                    ("u", (4,), "int32", "cache", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, t, u, y):
        with ft.For("i", 0, 4) as i:
            t[i] = x[i] + 1
            u[i] = t[i] + 1
            y[i] = u[i] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 4
    std = ft.pop_ast()

    assert std.match(ast)


def test_prop_across_loops():
    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("t", (4,), "int32", "cache", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, t, y):
        with ft.For("i", 0, 4) as i:
            t[i] = x[i] + 1
        with ft.For("i", 0, 4) as i:
            y[i] = t[i] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_used_in_many_stmts_no_prop():
    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[()] = x[i] + 1
                y1[i] = t[()] * 2
                y2[i] = t[()] * 3
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
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
    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[i] = x[i] + 1
                y[i] = t[i] * t[i] + t[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[i] = x[i] + 1
                y[i] = t[i] * t[i] + t[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_used_in_many_iterations_no_prop():
    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[i] = x[i] + 1
                with ft.For("j", 0, 8) as j:
                    y[i, j] = t[i] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
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
                t[i] = t[i] - 1
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
                t[i] = t[i] - 1
                y[i] = t[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_using_local_var_no_prop():
    with ft.VarDef([("x", (5, 10), "float64", "inout", "cpu"),
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

    with ft.VarDef([("x", (5, 10), "float64", "inout", "cpu"),
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
