import freetensor as ft


def test_cencel_const_in_numerator_and_denominator():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = (3 * x[()]) / (2 * y[()])
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = 1.5 * (x[()] / y[()])
    std = ft.pop_ast()

    assert std.match(ast)


def test_simplify_sqrt_1():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = ft.sqrt(x[()]) * ft.sqrt(x[()])
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = x[()]
    std = ft.pop_ast()

    assert std.match(ast)


def test_simplify_sqrt_2():
    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[()] = ft.min(2 * ft.sqrt(x1[()]), 3 * ft.sqrt(x2[()]))
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[()] = ft.sqrt(ft.min(4 * x1[()], 9 * x2[()]))
    std = ft.pop_ast()

    assert std.match(ast)


def test_simplify_sqrt_3():
    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[()] = ft.min(-2 * ft.sqrt(x1[()]), -3 * ft.sqrt(x2[()]))
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[()] = -1 * ft.sqrt(ft.max(4 * x1[()], 9 * x2[()]))
    std = ft.pop_ast()

    assert std.match(ast)


def test_simplify_sqrt_4():
    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, x3, y):
        y[()] = (ft.sqrt(x1[()]) * x2[()] /
                 ft.sqrt(x3[()])) * (ft.sqrt(x1[()]) * x2[()] / ft.sqrt(x3[()]))
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, x3, y):
        y[()] = ft.square(x2[()]) * (x1[()] / x3[()])
    std = ft.pop_ast()

    assert std.match(ast)


def test_simplify_square_abs():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = ft.abs(x[()]) * ft.abs(x[()])
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = ft.square(x[()])
    std = ft.pop_ast()

    assert std.match(ast)


def test_simplify_redundant_abs():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = ft.abs(ft.abs(x[()]))
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = ft.abs(x[()])
    std = ft.pop_ast()

    assert std.match(ast)


def test_fold_into_if_expr():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = x[()] * ft.if_then_else(x[()] > 0, 1, 0)
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = ft.if_then_else(x[()] > 0, x[()], 0)
    std = ft.pop_ast()

    assert std.match(ast)


def test_type_hint_from_user():
    with ft.VarDef([("x", (), "float32>=0", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = ft.abs(x[()])
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32>=0", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = x[()]
    std = ft.pop_ast()

    assert std.match(ast)
