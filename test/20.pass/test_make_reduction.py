import freetensor as ft


def test_reduce_sum():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] = y[...] + x[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] += x[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_reduce_prod():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[...] = 1
        with ft.For("i", 0, 4) as i:
            y[...] = y[...] * x[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[...] = 1
        with ft.For("i", 0, 4) as i:
            y[...] *= x[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_reduce_sub():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] = y[...] - x[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] -= x[i]
    std = ft.pop_ast()

    assert std.match(ast)
