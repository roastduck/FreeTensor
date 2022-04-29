import freetensor as ft


def test_basic():
    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("t", (4,), "int32", "cache", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, t, y):
        with ft.For("i", 0, 4) as i:
            t[i] = x[i] + 1
            y[i] = t[i] * 2
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_used_in_many_stmts_no_prop():
    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[()] = x[i] + 1
                y1[i] = t[()] * 2
                y2[i] = t[()] * 3
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[()] = x[i] + 1
                y1[i] = t[()] * 2
                y2[i] = t[()] * 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_used_in_many_reads_no_prop():
    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[i] = x[i] + 1
                y[i] = t[i] * t[i] + t[i]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[i] = x[i] + 1
                y[i] = t[i] * t[i] + t[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_modify_self_no_prop():
    with ft.VarDef([("x", (5,), "float64", "input", "cpu"),
                    ("y", (5,), "float64", "output", "cpu")]) as (x, y):
        with ft.VarDef("t", (5,), "float64", "cache", "cpu") as t:
            with ft.For("i", 0, 5) as i:
                t[i] = x[i]
            with ft.For("i", 0, 5) as i:
                t[i] = t[i] - 1
                y[i] = t[i]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([("x", (5,), "float64", "input", "cpu"),
                    ("y", (5,), "float64", "output", "cpu")]) as (x, y):
        with ft.VarDef("t", (5,), "float64", "cache", "cpu") as t:
            with ft.For("i", 0, 5) as i:
                t[i] = x[i]
            with ft.For("i", 0, 5) as i:
                t[i] = t[i] - 1
                y[i] = t[i]
    std = ft.pop_ast()

    assert std.match(ast)
