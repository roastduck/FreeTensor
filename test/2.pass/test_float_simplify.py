import ir


def test_simplify_sqrt_1():
    with ir.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = ir.sqrt(x[()]) * ir.sqrt(x[()])
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = x[()]
    std = ir.pop_ast()

    assert std.match(ast)


def test_simplify_sqrt_2():
    with ir.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[()] = ir.min(ir.sqrt(x1[()]), ir.sqrt(x2[()]))
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[()] = ir.sqrt(ir.min(x1[()], x2[()]))
    std = ir.pop_ast()

    assert std.match(ast)


def test_simplify_sqrt_3():
    with ir.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, x3, y):
        y[()] = (ir.sqrt(x1[()]) * x2[()] /
                 ir.sqrt(x3[()])) * (ir.sqrt(x1[()]) * x2[()] / ir.sqrt(x3[()]))
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, x3, y):
        y[()] = ir.square(x2[()]) * (x1[()] / x3[()])
    std = ir.pop_ast()

    assert std.match(ast)
