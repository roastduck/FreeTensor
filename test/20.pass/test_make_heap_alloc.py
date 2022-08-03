import freetensor as ft


def test_cpu_heap():
    with ft.VarDef([("x", (), "int32", "cache", "cpu"),
                    ("y", (2, 2), "int32", "cache", "cpu/heap"),
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

    with ft.VarDef([("x", (), "int32", "cache", "cpu"),
                    ("i", (), "int32", "input", "cpu"),
                    ("o", (), "int32", "output", "cpu")]) as (x, i, o):
        x[()] = i[()] * 2
        with ft.VarDef("y", (2, 2), "int32", "cache", "cpu/heap") as y:
            with ft.VarDef("t", (), "int32", "cache", "cpu/heap") as t:
                t[()] = x[()] + 1
                ft.Alloc(y)
                y[0, 1] = t[()] + 1
                y[1, 0] = t[()] + 1
            x[()] = y[0, 1] + y[1, 0] + 1
            ft.Free(y)
        o[()] = x[()] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_gpu_global_heap():
    with ft.Device(ft.GPU(), 0):
        with ft.VarDef([("x", (), "int32", "cache", "gpu/global"),
                        ("y", (2,), "int32", "cache", "gpu/global/heap"),
                        ("t", (), "int32", "cache", "gpu/global/heap"),
                        ("i", (), "int32", "input", "gpu/global"),
                        ("o", (), "int32", "output", "gpu/global")
                       ]) as (x, y, t, i, o):
            x[()] = i[()] * 2
            t[()] = x[()] + 1
            y[0] = t + 1
            y[1] = t + 1
            x[()] = y[0] + y[1] + 1
            o[()] = x[()] + 1
        ast = ft.pop_ast(verbose=True)
        ast = ft.lower(ast, verbose=1, skip_passes=["prop_one_time_use"])

        with ft.VarDef([("x", (), "int32", "cache", "gpu/global"),
                        ("i", (), "int32", "input", "gpu/global"),
                        ("o", (), "int32", "output", "gpu/global")]) as (x, i,
                                                                         o):
            x[()] = i[()] * 2
            with ft.VarDef("y", (2,), "int32", "cache", "gpu/global/heap") as y:
                with ft.VarDef("t", (), "int32", "cache",
                               "gpu/global/heap") as t:
                    t[()] = x[()] + 1
                    ft.Alloc(y)
                    y[0] = t[()] + 1
                    y[1] = t[()] + 1
                x[()] = y[0] + y[1] + 1
                ft.Free(y)
            o[()] = x[()] + 1
        std = ft.pop_ast()
        assert std.match(ast)


def test_transform_to_cpu_heap():
    with ft.VarDef([("x", (), "int32", "cache", "cpu"),
                    ("y", (2, 2), "int32", "cache", "cpu"),
                    ("t", (), "int32", "cache", "cpu"),
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

    with ft.VarDef([("x", (), "int32", "cache", "cpu"),
                    ("i", (), "int32", "input", "cpu"),
                    ("o", (), "int32", "output", "cpu")]) as (x, i, o):
        x[()] = i[()] * 2
        with ft.VarDef("y", (2, 2), "int32", "cache", "cpu/heap") as y:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[()] = x[()] + 1
                ft.Alloc(y)
                y[0, 1] = t[()] + 1
                y[1, 0] = t[()] + 1
            x[()] = y[0, 1] + y[1, 0] + 1
            ft.Free(y)
        o[()] = x[()] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_transform_to_gpu_global_heap():
    with ft.Device(ft.GPU(), 0):
        with ft.VarDef([("x", (), "int32", "cache", "gpu/global"),
                        ("y", (2,), "int32", "cache", "gpu/global"),
                        ("t", (), "int32", "cache", "gpu/global"),
                        ("i", (), "int32", "input", "gpu/global"),
                        ("o", (), "int32", "output", "gpu/global")
                       ]) as (x, y, t, i, o):
            x[()] = i[()] * 2
            t[()] = x[()] + 1
            y[0] = t + 1
            y[1] = t + 1
            x[()] = y[0] + y[1] + 1
            o[()] = x[()] + 1
        ast = ft.pop_ast(verbose=True)
        ast = ft.lower(ast, verbose=1, skip_passes=["prop_one_time_use"])

        with ft.VarDef([("x", (), "int32", "cache", "gpu/global"),
                        ("i", (), "int32", "input", "gpu/global"),
                        ("o", (), "int32", "output", "gpu/global")]) as (x, i,
                                                                         o):
            x[()] = i[()] * 2
            with ft.VarDef("y", (2,), "int32", "cache", "gpu/global/heap") as y:
                with ft.VarDef("t", (), "int32", "cache", "gpu/global") as t:
                    t[()] = x[()] + 1
                    ft.Alloc(y)
                    y[0] = t[()] + 1
                    y[1] = t[()] + 1
                x[()] = y[0] + y[1] + 1
                ft.Free(y)
            o[()] = x[()] + 1
        std = ft.pop_ast()
        assert std.match(ast)


def test_transform_dynamic_size():
    with ft.VarDef("n", (), "int32", "input", "cpu") as n:
        with ft.VarDef([("x", (n[...],), "int32", "input", "cpu"),
                        ("y", (n[...],), "int32", "cache", "cpu"),
                        ("s", (), "int32", "output", "cpu"),
                        ("m", (), "int32", "output", "cpu")]) as (x, y, s, m):
            s[...] = 0
            m[...] = 1
            with ft.For("i", 0, n[...]) as i:
                y[i] = x[i] * 2
            with ft.For("i", 0, n[...]) as i:
                s[...] += y[i]
                m[...] *= y[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("n", (), "int32", "input", "cpu") as n:
        with ft.VarDef([("x", (n[...],), "int32", "input", "cpu"),
                        ("s", (), "int32", "output", "cpu"),
                        ("m", (), "int32", "output", "cpu")]) as (x, s, m):
            s[...] = 0
            m[...] = 1
            with ft.VarDef("y", (n[...],), "int32", "cache", "cpu/heap") as y:
                ft.Alloc(y)
                with ft.For("i", 0, n[...]) as i:
                    y[i] = x[i] * 2
                with ft.For("i", 0, n[...]) as i:
                    s[...] += y[i]
                    m[...] *= y[i]
                ft.Free(y)
    std = ft.make_reduction(ft.pop_ast())
    assert std.match(ast)
