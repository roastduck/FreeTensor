import freetensor as ft
import pytest


def test_cpu_basic():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000, 1000), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 1000, nid="Li") as i:
            with ft.For("j", 0, 1000, nid="Lj") as j:
                y[i, j] = x[i, j] + 1

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.CPU())
    print(s.logs())
    assert s.logs() == ["merge(Li, Lj)", "parallelize(merged.Li.Lj, openmp)"]


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_basic_static_small():
    with ft.VarDef([("x", (10, 10, 2), "int32", "input", "cpu"),
                    ("y", (10, 10, 2), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 10, nid="Li") as i:
            with ft.For("j", 0, 10, nid="Lj") as j:
                y[i, j, 0] = x[i, j, 0] + 1

    device = ft.GPU()
    target = device.target()
    num_sm = target.multi_processor_count()

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(device)
    print(s.ast())
    print(s.logs())
    assert s.logs() == [
        "merge(Li, Lj)", f"split(merged.Li.Lj, -1, {num_sm}, 0)",
        "parallelize(merged.Li.Lj.0, blockIdx.x)",
        "parallelize(merged.Li.Lj.1, threadIdx.x)"
    ]


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_basic_static_large():
    with ft.VarDef([("x", (1000, 1000, 2), "int32", "input", "cpu"),
                    ("y", (1000, 1000, 2), "int32", "output", "cpu")]) as (x,
                                                                           y):
        with ft.For("i", 0, 1000, nid="Li") as i:
            with ft.For("j", 0, 1000, nid="Lj") as j:
                y[i, j, 0] = x[i, j, 0] + 1

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.GPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == [
        "merge(Li, Lj)", "split(merged.Li.Lj, 256, -1, 0)",
        "parallelize(merged.Li.Lj.0, blockIdx.x)",
        "parallelize(merged.Li.Lj.1, threadIdx.x)"
    ]


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_basic_dynamic():
    with ft.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ft.VarDef([("x", (n[()], 1000, 2), "int32", "input", "cpu"),
                        ("y", (n[()], 1000, 2), "int32", "output", "cpu")
                       ]) as (x, y):
            with ft.For("i", 0, n[()], nid="Li") as i:
                with ft.For("j", 0, 1000, nid="Lj") as j:
                    y[i, j, 0] = x[i, j, 0] + 1

    device = ft.GPU()
    target = device.target()
    num_sm = target.multi_processor_count()

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.GPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == [
        "merge(Li, Lj)", f"split(merged.Li.Lj, {num_sm}, -1, 0)",
        "reorder(merged.Li.Lj.1, merged.Li.Lj.0)",
        "split(merged.Li.Lj.0, 256, -1, 0)",
        "parallelize(merged.Li.Lj.1, blockIdx.y)",
        "parallelize(merged.Li.Lj.0.0, blockIdx.x)",
        "parallelize(merged.Li.Lj.0.1, threadIdx.x)"
    ]


def test_non_parallelizable():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 1000, nid="Li") as i:
            y[i] = 0
            with ft.For("j", 0, 1000, nid="Lj") as j:
                y[i] = y[i] * 2 + x[i, j]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.CPU())
    print(s.logs())
    assert s.logs() == ["parallelize(Li, openmp)"]


def test_reduction_better_not_parallelized():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 1000, nid="Li") as i:
            y[i] = 0
            with ft.For("j", 0, 1000, nid="Lj") as j:
                y[i] += x[i, j]

    ast = ft.make_reduction(ft.pop_ast())
    print(ast)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.CPU())
    print(s.logs())
    assert s.logs() == ["parallelize(Li, openmp)"]


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_warp_static():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "gpu/global"),
                    ("y", (1000, 2), "int32", "output", "gpu/global")]) as (x,
                                                                            y):
        with ft.For("i", 0, 1000, nid="Li") as i:
            y[i, 0] = 0
            with ft.For("k", 0, 1000, nid="Lk") as k:
                y[i, 0] += x[i, k]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.GPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == [
        "split(Lk, 32, -1, 0)", "parallelize(Lk.1, threadIdx.x)",
        "reorder(Lk.1, Lk.0)", "split(Li, 8, -1, 0)",
        "parallelize(Li.0, blockIdx.x)", "parallelize(Li.1, threadIdx.y)"
    ]


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_warp_dynamic():
    with ft.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ft.VarDef([("x", (n[()], n[()]), "int32", "input", "gpu/global"),
                        ("y", (n[()], 2), "int32", "output", "gpu/global")
                       ]) as (x, y):
            with ft.For("i", 0, n[()], nid="Li") as i:
                y[i, 0] = 0
                with ft.For("k", 0, n[()], nid="Lk") as k:
                    y[i, 0] += x[i, k]

    device = ft.GPU()
    target = device.target()
    num_sm = target.multi_processor_count()

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.GPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == [
        "split(Lk, 32, -1, 0)", "parallelize(Lk.1, threadIdx.x)",
        "reorder(Lk.1, Lk.0)", f"split(Li, {num_sm}, -1, 0)",
        "reorder(Li.1, Li.0)", "split(Li.0, 8, -1, 0)",
        "parallelize(Li.1, blockIdx.y)", "parallelize(Li.0.0, blockIdx.x)",
        "parallelize(Li.0.1, threadIdx.y)"
    ]


def test_outer_loop_too_short():
    with ft.VarDef([("x", (8, 1000), "int32", "input", "cpu"),
                    ("y", (8, 1000), "int32", "output", "cpu"),
                    ("z", (8, 1000), "int32", "output", "cpu")]) as (x, y, z):
        with ft.For("i", 0, 8, nid="Li") as i:
            with ft.For("j", 0, 1000, nid="Lj1") as j:
                y[i, j] = x[i, j] + x[i, (j + 1) % 1000]
            with ft.For("j", 0, 1000, nid="Lj2") as j:
                z[i, j] = y[i, j] + y[i, (j + 1) % 1000]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.CPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == ["parallelize(Lj1, openmp)", "parallelize(Lj2, openmp)"]


def test_outer_loop_not_parallelizable():
    with ft.VarDef([("x", (100,), "float32", "inout", "cpu"),
                    ("w", (100, 100), "float32", "input", "cpu")]) as (x, w):
        with ft.For("p", 0, 1000, nid="Lp") as p:
            with ft.VarDef("y", (100,), "float32", "cache", "cpu") as y:
                with ft.For("i", 0, 100, nid="Li0") as i:
                    y[i] = 0
                    with ft.For("j", 0, 100, nid="Lj") as j:
                        y[i] += x[j] * w[i, j]
                with ft.For("i", 0, 100, nid="Li1") as i:
                    x[i] = y[i]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.CPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == ["parallelize(Li0, openmp)", "parallelize(Li1, openmp)"]
