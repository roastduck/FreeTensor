import ir


def test_shorten_for():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 10) as i:
            with ir.If(i < 4):
                y[i] = x[i] * 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = x[i] * 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_side_effect_intrinsic():
    with ir.For("i", 0, 10) as i:
        with ir.If(i < 4):
            ir.Eval(ir.intrinsic("foobar()", has_side_effect=True))
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.For("i", 0, 4) as i:
        ir.Eval(ir.intrinsic("foobar()", has_side_effect=True))
    std = ir.pop_ast()

    assert std.match(ast)


def test_multiple_iterators():
    with ir.VarDef([("x", (766, 255, 255), "int32", "input", "cpu"),
                    ("y", (766, 255, 255), "int32", "output", "cpu")]) as (x,
                                                                           y):
        with ir.For("i", -10000, 10000) as i:
            with ir.For("j", 0, 256) as j:
                with ir.For("k", 0, 256) as k:
                    with ir.If(ir.l_and(i - j - k < 256, i - j - k >= 0)):
                        y[i, j, k] = x[i, j, k] * 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (766, 255, 255), "int32", "input", "cpu"),
                    ("y", (766, 255, 255), "int32", "output", "cpu")]) as (x,
                                                                           y):
        with ir.For("i", 0, 766) as i:
            with ir.For("j", ir.any(), ir.any()) as j:
                with ir.For("k", ir.any(), ir.any()) as k:
                    y[i, j, k] = x[i, j, k] * 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_linear_bounds():
    with ir.VarDef([("x", (11, 21), "int32", "input", "cpu"),
                    ("y", (11, 21), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 10000) as i:
            with ir.For("j", 0, 21) as j:
                with ir.If(ir.l_and(i - j <= 0, i + j <= 20)):
                    y[i, j] = x[i, j] * 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (11, 21), "int32", "input", "cpu"),
                    ("y", (11, 21), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 11) as i:
            with ir.For("j", ir.any(), ir.any()) as j:
                y[i, j] = x[i, j] * 2
    std = ir.pop_ast()

    assert std.match(ast)
