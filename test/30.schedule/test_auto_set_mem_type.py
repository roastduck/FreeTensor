import freetensor as ft
import pytest


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_basic():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "gpu/global"),
                    ("y", (1000, 1000), "int32", "output", "gpu/global")
                   ]) as (x, y):
        with ft.For("i", 0, 1000, nid="Li") as i:
            ft.MarkNid("V_t")
            with ft.VarDef("t", (), "int32", "cache", "gpu/global") as t:
                t[()] = 0
                with ft.For("j", 0, 1000, nid="Lj1") as j:
                    t[()] += x[i, j]
                with ft.For("j", 0, 1000, nid="Lj2") as j:
                    y[i, j] = x[i, j] - t[()]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.parallelize('Li', 'blockIdx.x')
    s.parallelize('Lj2', 'threadIdx.x')
    s.auto_set_mem_type(ft.GPU())
    print(s.ast())
    print(s.logs())
    assert s.logs()[2:] == ["set_mem_type(V_t, gpu/shared)"]


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_local_across_loops():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "gpu/global"),
                    ("y", (1000, 1000), "int32", "output", "gpu/global")
                   ]) as (x, y):
        with ft.For("i", 0, 1000, nid="Li") as i:
            ft.MarkNid("V_t")
            with ft.VarDef("t", (1000,), "int32", "cache", "gpu/global") as t:
                with ft.For("j", 0, 1000, nid="Lj1") as j:
                    t[j] = x[i, j] + 1
                with ft.For("j", 0, 1000, nid="Lj2") as j:
                    y[i, j] = t[j] * 2

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.parallelize('Li', 'blockIdx.x')
    s.parallelize('Lj1', 'threadIdx.x')
    s.parallelize('Lj2', 'threadIdx.x')
    s.auto_set_mem_type(ft.GPU())
    print(s.ast())
    print(s.logs())
    assert s.logs()[3:] == ["set_mem_type(V_t, gpu/local)"]
