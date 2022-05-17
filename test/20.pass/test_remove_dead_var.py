import freetensor as ft


def test_basic():
    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
        y[()] = x[()] + 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_chained():
    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
            with ft.VarDef("b", (), "int32", "cache", "cpu") as b:
                b[()] = a[()] + 1
                with ft.VarDef("c", (), "int32", "cache", "cpu") as c:
                    c[()] = a[()] + b[()]
        y[()] = x[()] + 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_self_assign():
    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = 0
            with ft.For("i", 0, 10) as i:
                a[()] = a[()] * x[()] + 1
        y[()] = x[()] + 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] + 1
    std = ft.pop_ast()

    assert std.match(ast)
