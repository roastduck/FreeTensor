import ir


def test_var_as_shape():

    @ir.transform
    def f(shape, x, y):
        ir.declare_var(shape, (2,), "int32", "input", "cpu")
        ir.declare_var(x, shape, "int32", "input", "cpu")
        ir.declare_var(y, shape, "int32", "output", "cpu")
        for i in range(shape[0]):
            for j in range(shape[1]):
                y[i, j] = x[i, j] * 2

    with ir.VarDef("shape", (2,), "int32", "input", "cpu") as shape:
        with ir.VarDef([("x", shape, "int32", "input", "cpu"),
                        ("y", shape, "int32", "output", "cpu")]) as (x, y):
            with ir.For("i", 0, shape[0]) as i:
                with ir.For("j", 0, shape[1]) as j:
                    y[i, j] = x[i, j] * 2
    assert ir.pop_ast().match(f.body)


def test_var_as_index():

    @ir.transform
    def f(idx, x, y):
        ir.declare_var(idx, (2,), "int32", "input", "cpu")
        ir.declare_var(x, (4, 4), "int32", "input", "cpu")
        ir.declare_var(y, (), "int32", "output", "cpu")
        y[()] = x[idx]

    with ir.VarDef([("idx", (2,), "int32", "input", "cpu"),
                    ("x", (4, 4), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (idx, x, y):
        y[()] = x[idx]
    assert ir.pop_ast().match(f.body)
