import freetensor as ft
import pytest


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
    ast = ft.lower(ast,
                   verbose=1,
                   target=ft.CPU().target(),
                   skip_passes=["prop_one_time_use"])

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


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_global_heap():
    with ft.GPU():
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
        ast = ft.lower(ast,
                       verbose=1,
                       target=ft.GPU(0).target(),
                       skip_passes=["prop_one_time_use"])

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
    ast = ft.lower(ast,
                   verbose=1,
                   target=ft.CPU().target(),
                   skip_passes=["prop_one_time_use"])

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


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_transform_to_gpu_global_heap():
    with ft.GPU():
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
        ast = ft.lower(ast,
                       verbose=1,
                       target=ft.GPU(0).target(),
                       skip_passes=["prop_one_time_use"])

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
    ast = ft.lower(ast,
                   verbose=1,
                   target=ft.CPU().target(),
                   skip_passes=['shrink_var'])

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
    std = ft.pop_ast()
    assert std.match(ast)


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_transform_dynamic_size_caused_by_thread_dim():
    with ft.VarDef([("x", (4,), "int32", "input", "gpu/global"),
                    ("s", (16,), "int32", "output", "gpu/global"),
                    ("m", (16,), "int32", "output", "gpu/global")]) as (x, s,
                                                                        m):
        with ft.For("i", 1, 4, label="Li") as i:
            with ft.For("j", 0, i, label="Lj") as j:
                with ft.VarDef("y", (), "int32", "cache", "gpu/global") as y:
                    y[...] = x[j] * i
                    s[i * 4 + j] = y[...] * 2
                    m[i * 4 + j] = y[...] * 3
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.parallelize("Lj", "threadIdx.x")
    ast = ft.lower(s.ast(),
                   verbose=1,
                   target=ft.GPU(0).target(),
                   skip_passes=['shrink_var'])

    with ft.VarDef([("x", (4,), "int32", "input", "gpu/global"),
                    ("s", (16,), "int32", "output", "gpu/global"),
                    ("m", (16,), "int32", "output", "gpu/global")]) as (x, s,
                                                                        m):
        with ft.For("i", 1, 4) as i:
            with ft.VarDef("y", (i,), "int32", "cache", "gpu/global/heap") as y:
                ft.Alloc(y)
                with ft.For("j", 0, i) as j:
                    y[j] = x[j] * i
                    s[i * 4 + j] = y[j] * 2
                    m[i * 4 + j] = y[j] * 3
                ft.Free(y)
    std = ft.pop_ast()
    assert std.match(ast)
