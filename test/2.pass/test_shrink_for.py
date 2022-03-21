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
