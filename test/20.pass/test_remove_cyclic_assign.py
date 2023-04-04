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


def test_in_branch():
    with ft.VarDef([("cond", (), "bool", "input", "cpu"),
                    ("a", (), "int32", "inout", "cpu"),
                    ("b", (), "int32", "output", "cpu")]) as (cond, a, b):
        with ft.If(cond[()]):
            b[()] = a[()]
            a[()] = b[()]
    ast = ft.pop_ast(verbose=True)
    # Run this pass only. No sinking vars into If
    ast = ft.remove_cyclic_assign(ast)
    print(ast)

    with ft.VarDef([("cond", (), "bool", "input", "cpu"),
                    ("a", (), "int32", "inout", "cpu"),
                    ("b", (), "int32", "output", "cpu")]) as (cond, a, b):
        with ft.If(cond[()]):
            b[()] = a[()]
    std = ft.pop_ast()

    assert std.match(ast)
