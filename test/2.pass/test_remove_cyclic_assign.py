import ir


def test_basic():
    with ir.VarDef([("a", (), "int32", "inout", "cpu"),
                    ("b", (), "int32", "output", "cpu")]) as (a, b):
        b[()] = a[()]
        a[()] = b[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("a", (), "int32", "inout", "cpu"),
                    ("b", (), "int32", "output", "cpu")]) as (a, b):
        b[()] = a[()]
    std = ir.pop_ast()

    assert std.match(ast)
