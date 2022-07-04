import freetensor as ft


def test_basic():
    with ft.VarDef([("x", (), "int32", "cache", "cpu"),
                    ("y", (2,2,), "int32", "cache", "cpu/heap"),
                    ("t", (), "int32", "cache", "cpu/heap"),
                    ("i", (), "int32", "input", "cpu"),
                    ("o", (), "int32", "output", "cpu")]) as (x, y, t, i, o):
        x[()] = i[()] * 2
        t[()] = x[()] + 1
        y[0, 1] = t + 1
        y[1, 0] = t + 1
        x[()] = y[0, 1] + y[1, 0] + 1
        o[()] = x[()] + 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=["prop_one_time_use"])
    
    # func = ft.lower(ft.Func("main", ["i"], [], ft.pop_ast()), verbose=1, skip_passes=["prop_one_time_use"])
    # code = ft.codegen(func, verbose=True)

    with ft.VarDef([("x", (), "int32", "cache", "cpu"),
                    ("i", (), "int32", "input", "cpu"),
                    ("o", (), "int32", "output", "cpu")]) as (x, i, o):
        x[()] = i[()] * 2
        with ft.VarDef("y", (2,2,), "int32", "cache", "cpu/heap") as y:
            with ft.VarDef("t", (), "int32", "cache", "cpu/heap") as t:
                t[()] = x[()] + 1
                ft.Alloc(y.name)
                y[0, 1] = t[()] + 1
                y[1, 0] = t[()] + 1
            x[()] = y[0, 1] + y[1, 0] + 1
            ft.Free(y.name)
        o[()] = x[()] + 1
    std = ft.pop_ast()

    assert std.match(ast)