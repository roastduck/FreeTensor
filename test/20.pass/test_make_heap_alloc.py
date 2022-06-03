import freetensor as ft


def test_basic():
    with ft.VarDef([("x", (), "int32", "cache", "cpu/heap"),
                    ("i", (), "int32", "input", "cpu"),
                    ("o", (), "int32", "output", "cpu")]) as (x, i, o):
        with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
            t[()] = i[()] * 7
            x[()] = t[()] + 2
        o[()] = x[()] * 2
        o[()] *= 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("i", (), "int32", "input", "cpu"),
                    ("o", (), "int32", "output", "cpu")]) as (i, o):
        o[()] = 28 * i[()] + 8
    std = ft.pop_ast()

    assert std.match(ast)

# def test_basic_origin():
#     with ft.VarDef([("x", (), "int32", "inout", "cpu"),
#                     ("y", (), "int32", "output", "cpu")]) as (x, y):
#         with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
#             a[()] = x[()] + 1
#         y[()] = x[()] + 1
#     ast = ft.pop_ast(verbose=True)
#     ast = ft.lower(ast, verbose=1)

#     with ft.VarDef([("x", (), "int32", "inout", "cpu"),
#                     ("y", (), "int32", "output", "cpu")]) as (x, y):
#         y[()] = x[()] + 1
#     std = ft.pop_ast()

#     assert std.match(ast)