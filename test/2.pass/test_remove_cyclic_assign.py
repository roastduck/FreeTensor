import freetensor as ft


def test_basic():
    with ft.VarDef([("a", (), "int32", "inout", "cpu"),
                    ("b", (), "int32", "output", "cpu")]) as (a, b):
        b[()] = a[()]
        a[()] = b[()]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([("a", (), "int32", "inout", "cpu"),
                    ("b", (), "int32", "output", "cpu")]) as (a, b):
        b[()] = a[()]
    std = ft.pop_ast()

    assert std.match(ast)
