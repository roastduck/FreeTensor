import freetensor as ft


def test_basic():
    with ft.VarDef([("a", (), "int32", "inout", "cpu"),
                    ("b", (), "int32", "output", "cpu")]) as (a, b):
        b[()] = a[()]
        a[()] = b[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("a", (), "int32", "inout", "cpu"),
                    ("b", (), "int32", "output", "cpu")]) as (a, b):
        b[()] = a[()]
    std = ft.pop_ast()

    assert std.match(ast)
