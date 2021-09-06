import ir


def test_basic():
    with ir.VarDef([
        ("x1", (), "int32", "input", "cpu"),
        ("x2", (), "int32", "input", "cpu"),
        ("x3", (), "int32", "input", "cpu"),
        ("y", (), "int32", "output", "cpu"),
    ]) as (x1, x2, x3, y):
        y[()] = (x1[()] + x2[()]) * x3[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.grad(ast, {"x1": "d_x1", "x2": "d_x2", "x3": "d_x3", "y": "d_y"})
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("d_x1", (), "int32", "output", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("d_x2", (), "int32", "output", "cpu"),
                    ("x3", (), "int32", "input", "cpu"),
                    ("d_x3", (), "int32", "output", "cpu"),
                    ("y", (), "int32", "input", "cpu"),
                    ("d_y", (), "int32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        with ir.VarDef("d_y.old", (), "int32", "cache", "cpu") as d_y_old:
            d_y_old[()] = d_y[()]
            d_y[()] = 0
            d_x1[()] = 0 + d_y_old[()] * x3[()]
            d_x2[()] = 0 + d_y_old[()] * x3[()]
            d_x3[()] = 0 + d_y_old[()] * (x1[()] + x2[()])
    std = ir.pop_ast()

    assert std.match(ast)


def test_multiple_statements():
    with ir.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("x3", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x1, x2, x3, y):
        with ir.VarDef("t", (), "int32", "cache", "cpu") as t:
            t[()] = x1[()] + x2[()]
            y[()] = t[()] * x3[()]
    ast = ir.pop_ast()
    print(ast)
    # TODO: Derive t -> d_t automatically
    ast = ir.grad(ast, {
        "x1": "d_x1",
        "x2": "d_x2",
        "x3": "d_x3",
        "y": "d_y",
        "t": "d_t"
    })
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("d_x1", (), "int32", "output", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("d_x2", (), "int32", "output", "cpu"),
                    ("x3", (), "int32", "input", "cpu"),
                    ("d_x3", (), "int32", "output", "cpu"),
                    ("y", (), "int32", "input", "cpu"),
                    ("d_y", (), "int32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        with ir.VarDef([("t", (), "int32", "cache", "cpu"),
                        ("d_t", (), "int32", "cache", "cpu")]) as (t, d_t):
            t[()] = x1[()] + x2[()]
            with ir.VarDef("d_y.old", (), "int32", "cache", "cpu") as d_y_old:
                d_y_old[()] = d_y[()]
                d_y[()] = 0
                d_t[()] = 0 + d_y_old[()] * x3[()]
                d_x3[()] = 0 + d_y_old[()] * t[()]
            with ir.VarDef("d_t.old", (), "int32", "cache", "cpu") as d_t_old:
                d_t_old[()] = d_t[()]
                d_t[()] = 0
                d_x1[()] = d_t_old[()]
                d_x2[()] = d_t_old[()]
    std = ir.pop_ast()

    assert std.match(ast)


def test_dependent_iterations():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[()] = -y[()] + x[i]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.grad(ast, {"x": "d_x", "y": "d_y"})
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("d_x", (4,), "int32", "output", "cpu")]) as (x, d_x):
        with ir.For("i1", 0, 4) as i:
            d_x[i] = 0
        with ir.VarDef([("y", (), "int32", "input", "cpu"),
                        ("d_y", (), "int32", "inout", "cpu")]) as (y, d_y):
            with ir.For("i", 0, 4) as i:
                with ir.VarDef("d_y.old", (), "int32", "cache",
                               "cpu") as d_y_old:
                    d_y_old[()] = d_y[()]
                    d_y[()] = -1 * d_y_old[()]
                    d_x[-1 * i + 4] += d_y_old[()]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)
