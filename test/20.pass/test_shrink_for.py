import freetensor as ft


def test_shorten_for():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 10) as i:
            with ft.If(i < 4):
                y[i] = x[i] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x[i] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_side_effect_intrinsic():
    with ft.For("i", 0, 10) as i:
        with ft.If(i < 4):
            ft.Eval(ft.intrinsic("foobar()", has_side_effect=True))
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.For("i", 0, 4) as i:
        ft.Eval(ft.intrinsic("foobar()", has_side_effect=True))
    std = ft.pop_ast()

    assert std.match(ast)


def test_multiple_iterators():
    with ft.VarDef([("x", (766, 255, 255), "int32", "input", "cpu"),
                    ("y", (766, 255, 255), "int32", "output", "cpu")]) as (x,
                                                                           y):
        with ft.For("i", -10000, 10000) as i:
            with ft.For("j", 0, 256) as j:
                with ft.For("k", 0, 256) as k:
                    with ft.If(ft.l_and(i - j - k < 256, i - j - k >= 0)):
                        y[i, j, k] = x[i, j, k] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (766, 255, 255), "int32", "input", "cpu"),
                    ("y", (766, 255, 255), "int32", "output", "cpu")]) as (x,
                                                                           y):
        with ft.For("i", 0, 766) as i:
            with ft.For("j", ft.any(), ft.any()) as j:
                with ft.For("k", ft.any(), ft.any()) as k:
                    y[i, j, k] = x[i, j, k] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_linear_bounds():
    with ft.VarDef([("x", (11, 21), "int32", "input", "cpu"),
                    ("y", (11, 21), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 10000) as i:
            with ft.For("j", 0, 21) as j:
                with ft.If(ft.l_and(i - j <= 0, i + j <= 20)):
                    y[i, j] = x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (11, 21), "int32", "input", "cpu"),
                    ("y", (11, 21), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 11) as i:
            with ft.For("j", ft.any(), ft.any()) as j:
                y[i, j] = x[i, j] * 2
    std = ft.pop_ast()

    assert std.match(ast)
