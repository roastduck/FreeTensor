import freetensor as ft
import pytest


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_basic():
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
    s = ft.Schedule(ast)
    s.parallelize('Li', 'blockIdx.x')
    s.parallelize('Lj2', 'threadIdx.x')
    s.auto_set_mem_type(ft.GPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert logs[2:] == ["set_mem_type(V_t, gpu/shared)"]


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_local_across_loops():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "gpu/global"),
                    ("y", (1000, 1000), "int32", "output", "gpu/global")
                   ]) as (x, y):
        with ft.For("i", 0, 1000, label="Li") as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (1000,), "int32", "cache", "gpu/global") as t:
                with ft.For("j", 0, 1000, label="Lj1") as j:
                    t[j] = x[i, j] + 1
                with ft.For("j", 0, 1000, label="Lj2") as j:
                    y[i, j] = t[j] * 2

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.parallelize('Li', 'blockIdx.x')
    s.parallelize('Lj1', 'threadIdx.x')
    s.parallelize('Lj2', 'threadIdx.x')
    s.auto_set_mem_type(ft.GPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert logs[3:] == ["set_mem_type(V_t, gpu/local)"]


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_no_too_large_shared():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "gpu/global"),
                    ("y", (1000, 1000), "int32", "output", "gpu/global")
                   ]) as (x, y):
        ft.MarkLabel("V_v")
        with ft.VarDef("v", (1000,), "int32", "cache", "gpu/global") as v:
            # v: 1 threads * 1000 * sizeof(int32) = 4kB, not too large
            with ft.For("i", 0, 1000, label="Li") as i:  # threadIdx.y
                ft.MarkLabel("V_t")
                with ft.VarDef("t", (1000,), "int32", "cache",
                               "gpu/global") as t:
                    ft.MarkLabel("V_u")
                    with ft.VarDef("u", (), "int32", "cache",
                                   "gpu/global") as u:
                        # t: 1000 threads * 1000 * sizeof(int32) = 4MB, too large
                        # u: 1000 threads * 1 * sizeof(int32) = 4kB, not too large
                        u[...] = 0
                        with ft.For("j", 0, 1000, label="Lj1") as j:
                            t[j] = x[i, j] + x[i, (j + 1) % 1000]
                            u[...] += x[i, j]
                        v[i] = x[i, 0] + x[(i + 1) % 1000, 0]
                        with ft.For("j", 0, 1000, label="Lj2") as j:
                            y[i, j] = x[i, j] * t[j] * u[...] * v[i]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.parallelize('Li', 'threadIdx.y')
    s.parallelize('Lj2', 'threadIdx.x')
    s.auto_set_mem_type(ft.GPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert sorted(logs[2:]) == sorted(
        ["set_mem_type(V_u, gpu/shared)", "set_mem_type(V_v, gpu/shared)"])
