import freetensor as ft


def test_reduce_add():
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
            y[...] += -1 * x[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_reduce_add_sub_1():
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] = (y[...] - x1[i]) - x2[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] += -1 * (x1[i] + x2[i])
    std = ft.pop_ast()

    assert std.match(ast)


def test_reduce_add_sub_2():
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] = (y[...] - x1[i]) + x2[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] += x2[i] - x1[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_reduce_add_sub_3():
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] = y[...] - (x1[i] - x2[i])
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] += x2[i] - x1[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_reduce_add_sub_4():
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] = y[...] - (x1[i] + x2[i])
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] += -1 * (x1[i] + x2[i])
    std = ft.pop_ast()

    assert std.match(ast)
