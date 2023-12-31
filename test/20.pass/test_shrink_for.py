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


def test_presburger_bounds():
    with ft.VarDef([("x", (128, 128), "int32", "input", "cpu"),
                    ("y", (128, 128), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i0", 0, 8) as i0:
            with ft.For("j0", 0, 8) as j0:
                with ft.For("i", 0, 128) as i:
                    with ft.For("j", 0, 128) as j:
                        with ft.If(ft.l_and(i // 16 == i0, j // 16 == j0)):
                            y[i, j] = x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (128, 128), "int32", "input", "cpu"),
                    ("y", (128, 128), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i0", 0, 8) as i0:
            with ft.For("j0", 0, 8) as j0:
                with ft.For("i", 16 * i0, 16 * i0 + 16) as i:
                    with ft.For("j", 16 * j0, 16 * j0 + 16) as j:
                        y[i, j] = x[i, j] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_presburger_bounds_strided():
    with ft.VarDef([("x", (128, 128), "int32", "input", "cpu"),
                    ("y", (128, 128), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i0", 0, 8) as i0:
            with ft.For("j0", 0, 8) as j0:
                with ft.For("i", 0, 128) as i:
                    with ft.For("j", 0, 128) as j:
                        with ft.If(ft.l_and(i % 8 == i0, j % 8 == j0)):
                            y[i, j] = x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (128, 128), "int32", "input", "cpu"),
                    ("y", (128, 128), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i0", 0, 8) as i0:
            with ft.For("j0", 0, 8) as j0:
                with ft.For("i", i0, i0 + 121, 8) as i:
                    with ft.For("j", j0, j0 + 121, 8) as j:
                        y[i, j] = x[i, j] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_multiple_branches():
    with ft.VarDef([("x", (32, 4), "int32", "inout", "cpu"),
                    ("y", (32, 4), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 32) as i:
            with ft.For("j", 0, 4, label='L') as j:
                with ft.If((j // 2) * 32 + i < 50):
                    with ft.If(j % 2 == 0):
                        y[i, j] = x[i, j] * 2
                    y[i, j] += 1
    ast1 = ft.pop_ast(verbose=True)
    ast1 = ft.shrink_for(ast1)
    print(ast1)

    with ft.VarDef([("x", (32, 4), "int32", "inout", "cpu"),
                    ("y", (32, 4), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 32) as i:
            with ft.For("j", 0, 4, label='L') as j:
                with ft.If((j // 2) * 32 + i < 50):
                    y[i, j] += 1
    ast2 = ft.pop_ast(verbose=True)
    ast2 = ft.shrink_for(ast2)
    print(ast2)

    # No matter if the `if j % 2 == 0` branch exists, the shrinked length to be
    # the same
    assert ft.find_stmt(ast1, 'L').len.same_as(ft.find_stmt(ast2, 'L').len)
