import freetensor as ft


def test_chained_subscript():

    @ft.transform
    def f(x, y):
        x: ft.Var((4, 4), "int32", "input", "cpu")
        y: ft.Var((4, 4), "int32", "output", "cpu")
        for i in range(4):
            for j in range(4):
                y[i][j] = x[i][j] * 2

    print(f)

    with ft.VarDef([("x", (4, 4), "int32", "input", "cpu"),
                    ("y", (4, 4), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 4) as j:
                y[i, j] = x[i, j] * 2
    assert ft.pop_ast().match(f.body)


def test_select():

    @ft.transform
    def f(x, y):
        x: ft.Var((4, 4), "int32", "input", "cpu")
        y: ft.Var((4, 4), "int32", "output", "cpu")
        for i in range(4):
            for j in range(4):
                y.select(j, 1).select(i,
                                      0)[()] = x.select(i, 0).select(j, 0) * 2

    f = ft.simplify_pass(f)
    print(f)

    with ft.VarDef([("x", (4, 4), "int32", "input", "cpu"),
                    ("y", (4, 4), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 4) as j:
                y[i, j] = x[i, j] * 2
    assert ft.pop_ast().match(f.body)


def test_var_as_shape():

    @ft.transform
    def f(shape, x, y):
        shape: ft.Var((2,), "int32", "input", "cpu")
        x: ft.Var(shape, "int32", "input", "cpu")
        y: ft.Var(shape, "int32", "output", "cpu")
        for i in range(shape[0]):
            for j in range(shape[1]):
                y[i, j] = x[i, j] * 2

    with ft.VarDef("shape", (2,), "int32", "input", "cpu") as shape:
        with ft.VarDef([("x", shape, "int32", "input", "cpu"),
                        ("y", shape, "int32", "output", "cpu")]) as (x, y):
            with ft.For("i", 0, shape[0]) as i:
                with ft.For("j", 0, shape[1]) as j:
                    y[i, j] = x[i, j] * 2
    assert ft.pop_ast().match(f.body)


def test_var_as_index():

    @ft.transform
    def f(idx, x, y):
        idx: ft.Var((2,), "int32", "input", "cpu")
        x: ft.Var((4, 4), "int32", "input", "cpu")
        y: ft.Var((), "int32", "output", "cpu")
        y[()] = x[idx]
        # TODO: Consider x[*idx]

    with ft.VarDef([("idx", (2,), "int32", "input", "cpu"),
                    ("x", (4, 4), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (idx, x, y):
        y[()] = x[idx]
    assert ft.pop_ast().match(f.body)


def test_var_as_index_2():

    @ft.transform
    def f(idx, x, y):
        idx: ft.Var((2,), "int32", "input", "cpu")
        x: ft.Var((4, 4), "int32", "input", "cpu")
        y: ft.Var((), "int32", "output", "cpu")
        y[()] = x[idx[0], idx[1]]

    with ft.VarDef([("idx", (2,), "int32", "input", "cpu"),
                    ("x", (4, 4), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (idx, x, y):
        y[()] = x[idx]
    assert ft.pop_ast().match(f.body)
