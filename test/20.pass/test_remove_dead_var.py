import freetensor as ft


def test_basic():
    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
        y[()] = x[()] + 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_chained():
    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
            with ft.VarDef("b", (), "int32", "cache", "cpu") as b:
                b[()] = a[()] + 1
                with ft.VarDef("c", (), "int32", "cache", "cpu") as c:
                    c[()] = a[()] + b[()]
        y[()] = x[()] + 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_self_assign():
    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = 0
            with ft.For("i", 0, 10) as i:
                a[()] = a[()] * x[()] + 1
        y[()] = x[()] + 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_remove_a_write_if_no_reads_after_it():
    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
            y[()] = a[()] * a[()]
            a[()] *= 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
            y[()] = a[()] * a[()]
    std = ft.pop_ast()

    assert std.match(ast)


def test_remove_a_write_in_a_loop_if_no_reads_after_it():
    with ft.VarDef([("x", (4,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (4,), "int32", "cache", "cpu") as a:
            with ft.For("i", 0, 4) as i:
                a[i] = x[i] + 1
            with ft.For("i", 0, 4) as i:
                y[i] = a[i] * a[i]
            with ft.For("i", 0, 4) as i:
                a[i] *= 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (4,), "int32", "cache", "cpu") as a:
            with ft.For("i", 0, 4) as i:
                a[i] = x[i] + 1
            with ft.For("i", 0, 4) as i:
                y[i] = a[i] * a[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_no_remove_writes_if_maybe_looped_around():
    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (2,), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
            with ft.For("i", 0, 2) as i:
                y[i] = a[()] * a[()]
                a[()] *= 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (2,), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
            with ft.For("i", 0, 2) as i:
                y[i] = a[()] * a[()]
                a[()] *= 2
    std = ft.pop_ast()

    assert std.match(ast)
