import freetensor as ft
import pytest


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_basic():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "gpu/global"),
                    ("y", (1000, 1000), "int32", "output", "gpu/global")
                   ]) as (x, y):
        with ft.For("i", 0, 1000, label="Li") as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (), "int32", "cache", "gpu/global") as t:
                t[()] = 0
                with ft.For("j", 0, 1000, label="Lj1") as j:
                    t[()] += x[i, j]
                with ft.For("j", 0, 1000, label="Lj2") as j:
                    y[i, j] = x[i, j] - t[()]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=1)
    s.parallelize('Li', 'blockIdx.x')
    s.set_mem_type('V_t', 'gpu/local')

    with ft.VarDef([("x", (1000, 1000), "int32", "input", "gpu/global"),
                    ("y", (1000, 1000), "int32", "output", "gpu/global")
                   ]) as (x, y):
        with ft.For("i", 0, 1000, label="Li") as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (), "int32", "cache", "gpu/local") as t:
                t[()] = 0
                with ft.For("j", 0, 1000, label="Lj1") as j:
                    t[()] += x[i, j]
                with ft.For("j", 0, 1000, label="Lj2") as j:
                    y[i, j] = x[i, j] - t[()]
    std = ft.pop_ast()

    assert std.match(s.ast())


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_not_found():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "gpu/global"),
                    ("y", (1000, 1000), "int32", "output", "gpu/global")
                   ]) as (x, y):
        with ft.For("i", 0, 1000, label="Li") as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (), "int32", "cache", "gpu/global") as t:
                t[()] = 0
                with ft.For("j", 0, 1000, label="Lj1") as j:
                    t[()] += x[i, j]
                with ft.For("j", 0, 1000, label="Lj2") as j:
                    y[i, j] = x[i, j] - t[()]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=1)
    s.parallelize('Li', 'blockIdx.x')
    with pytest.raises(ft.InvalidSchedule):
        s.set_mem_type('XXXX', 'gpu/local')


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_reject_indicet_access_by_load():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "gpu/global"),
                    ("y", (1000, 1000), "int32", "output", "gpu/global"),
                    ("offset", (), "int32", "input", "byvalue")]) as (x, y,
                                                                      offset):
        with ft.For("i", 0, 1000, label="Li") as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (10,), "int32", "cache", "gpu/global") as t:
                t[offset[...]] = 0
                with ft.For("j", 0, 1000, label="Lj1") as j:
                    t[offset[...]] += x[i, j]
                with ft.For("j", 0, 1000, label="Lj2") as j:
                    y[i, j] = x[i, j] - t[offset[...]]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=1)
    s.parallelize('Li', 'blockIdx.x')
    with pytest.raises(ft.InvalidSchedule):
        s.set_mem_type('V_t', 'gpu/local')


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_reject_indicet_access_dynamic_loop():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "gpu/global"),
                    ("y", (1000, 1000), "int32", "output", "gpu/global"),
                    ("n", (), "int32", "input", "byvalue")]) as (x, y, n):
        with ft.For("i", 0, 1000, label="Li") as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (n[...],), "int32", "cache", "gpu/global") as t:
                with ft.For("j", 0, n[...], label="Lj1") as j:
                    t[j] = x[i, j] + 1
                with ft.For("j", 0, n[...], label="Lj2") as j:
                    y[i, j] = t[j] * t[j]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=1)
    s.parallelize('Li', 'blockIdx.x')
    with pytest.raises(ft.InvalidSchedule):
        s.set_mem_type('V_t', 'gpu/local')
