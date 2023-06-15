import fnmatch

import freetensor as ft
import pytest


def fnmatch_list(strings, patterns):
    if len(patterns) != len(strings):
        return False
    for string, pattern in zip(strings, patterns):
        if not fnmatch.fnmatch(string, pattern):
            return False
    return True


def test_cpu_basic():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000, 1000), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 1000, label="Li") as i:
            with ft.For("j", 0, 1000, label="Lj") as j:
                y[i, j] = x[i, j] + 1

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.CPU())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(
        logs, ["merge(Li, Lj)", "parallelize($merge{Li, Lj}, openmp, *)"])


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_basic_static_small():
    with ft.VarDef([("x", (10, 10, 2), "int32", "input", "cpu"),
                    ("y", (10, 10, 2), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 10, label="Li") as i:
            with ft.For("j", 0, 10, label="Lj") as j:
                y[i, j, 0] = x[i, j, 0] + 1

    device = ft.GPU()
    target = device.target()
    num_sm = target.multi_processor_count()

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(device)
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(logs, [
        "merge(Li, Lj)", f"split($merge{{Li, Lj}}, -1, {num_sm}, 0)",
        "parallelize($split.0{$merge{Li, Lj}}, blockIdx.x, *)",
        "parallelize($split.1{$merge{Li, Lj}}, threadIdx.x, *)"
    ])


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_basic_static_large():
    with ft.VarDef([("x", (1000, 1000, 2), "int32", "input", "cpu"),
                    ("y", (1000, 1000, 2), "int32", "output", "cpu")]) as (x,
                                                                           y):
        with ft.For("i", 0, 1000, label="Li") as i:
            with ft.For("j", 0, 1000, label="Lj") as j:
                y[i, j, 0] = x[i, j, 0] + 1

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.GPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(logs, [
        "merge(Li, Lj)", "split($merge{Li, Lj}, 256, -1, 0)",
        "parallelize($split.0{$merge{Li, Lj}}, blockIdx.x, *)",
        "parallelize($split.1{$merge{Li, Lj}}, threadIdx.x, *)"
    ])


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_basic_dynamic():
    with ft.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ft.VarDef([("x", (n[()], 1000, 2), "int32", "input", "cpu"),
                        ("y", (n[()], 1000, 2), "int32", "output", "cpu")
                       ]) as (x, y):
            with ft.For("i", 0, n[()], label="Li") as i:
                with ft.For("j", 0, 1000, label="Lj") as j:
                    y[i, j, 0] = x[i, j, 0] + 1

    device = ft.GPU()
    target = device.target()
    num_sm = target.multi_processor_count()

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.GPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(logs, [
        "merge(Li, Lj)", f"split($merge{{Li, Lj}}, {num_sm}, -1, 0)",
        "reorder($split.1{$merge{Li, Lj}}, $split.0{$merge{Li, Lj}})",
        "split($split.0{$merge{Li, Lj}}, 256, -1, 0)",
        "parallelize($split.1{$merge{Li, Lj}}, blockIdx.y, *)",
        "parallelize($split.0{$split.0{$merge{Li, Lj}}}, blockIdx.x, *)",
        "parallelize($split.1{$split.0{$merge{Li, Lj}}}, threadIdx.x, *)"
    ])


def test_non_parallelizable():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 1000, label="Li") as i:
            y[i] = 0
            with ft.For("j", 0, 1000, label="Lj") as j:
                y[i] = y[i] * 2 + x[i, j]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.CPU())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(logs, ["parallelize(Li, openmp, *)"])


def test_reduction_unable_to_parallelize():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 1000, label="Li") as i:
            y[i] = 0  # Initializing here stops paralization
            with ft.For("j", 0, 1000, label="Lj") as j:
                y[i] += x[i, j]

    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.CPU())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(logs, ["parallelize(Li, openmp, *)"])


@pytest.mark.skipif(
    ft.CPU().target().n_cores() < 16 or ft.CPU().target().n_cores() > 1000,
    reason="This test is designed for systems with typical number of cores")
def test_reduction_better_not_parallelized():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 1000, label="Linit") as i:
            y[i] = 0
        with ft.For("i", 0, 1000, label="Li") as i:
            with ft.For("j", 0, 1000, label="Lj") as j:
                y[i] += x[i, j]

    print(f"There are {ft.CPU().target().n_cores()} cores")
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.CPU())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(
        logs, ["parallelize(Linit, openmp, *)", "parallelize(Li, openmp, *)"])


@pytest.mark.skipif(
    ft.CPU().target().n_cores() < 16 or ft.CPU().target().n_cores() > 1000,
    reason="This test is designed for systems with typical number of cores")
def test_reduction_better_parallelized():
    with ft.VarDef([("x", (4, 4), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="Linit") as i:
            y[i] = 0
        with ft.For("i", 0, 4, label="Li") as i:
            with ft.For("j", 0, 4, label="Lj") as j:
                y[i] += x[i, j]

    print(f"There are {ft.CPU().target().n_cores()} cores")
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.CPU())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(logs, [
        'parallelize(Linit, openmp, *)', 'merge(Li, Lj)',
        'parallelize($merge{Li, Lj}, openmp, *)'
    ])


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_warp_static():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "gpu/global"),
                    ("y", (1000, 2), "int32", "output", "gpu/global")]) as (x,
                                                                            y):
        with ft.For("i", 0, 1000, label="Li") as i:
            y[i, 0] = 0
            with ft.For("k", 0, 1000, label="Lk") as k:
                y[i, 0] += x[i, k]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.GPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(logs, [
        "split(Lk, 32, -1, 0)", "parallelize($split.1{Lk}, threadIdx.x, *)",
        "reorder($split.1{Lk}, $split.0{Lk})", "split(Li, 8, -1, 0)",
        "parallelize($split.0{Li}, blockIdx.x, *)",
        "parallelize($split.1{Li}, threadIdx.y, *)"
    ])


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_warp_dynamic():
    with ft.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ft.VarDef([("x", (n[()], n[()]), "int32", "input", "gpu/global"),
                        ("y", (n[()], 2), "int32", "output", "gpu/global")
                       ]) as (x, y):
            with ft.For("i", 0, n[()], label="Li") as i:
                y[i, 0] = 0
                with ft.For("k", 0, n[()], label="Lk") as k:
                    y[i, 0] += x[i, k]

    device = ft.GPU()
    target = device.target()
    num_sm = target.multi_processor_count()

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.GPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(logs, [
        "split(Lk, 32, -1, 0)", "parallelize($split.1{Lk}, threadIdx.x, *)",
        "reorder($split.1{Lk}, $split.0{Lk})", f"split(Li, {num_sm}, -1, 0)",
        "reorder($split.1{Li}, $split.0{Li})", "split($split.0{Li}, 8, -1, 0)",
        "parallelize($split.1{Li}, blockIdx.y, *)",
        "parallelize($split.0{$split.0{Li}}, blockIdx.x, *)",
        "parallelize($split.1{$split.0{Li}}, threadIdx.y, *)"
    ])


def test_outer_loop_too_short():
    with ft.VarDef([("x", (8, 1000), "int32", "input", "cpu"),
                    ("y", (8, 1000), "int32", "output", "cpu"),
                    ("z", (8, 1000), "int32", "output", "cpu")]) as (x, y, z):
        with ft.For("i", 0, 8, label="Li") as i:
            with ft.For("j", 0, 1000, label="Lj1") as j:
                y[i, j] = x[i, j] + x[i, (j + 1) % 1000]
            with ft.For("j", 0, 1000, label="Lj2") as j:
                z[i, j] = y[i, j] + y[i, (j + 1) % 1000]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(
        logs, ["parallelize(Lj1, openmp, *)", "parallelize(Lj2, openmp, *)"])


def test_outer_loop_not_parallelizable():
    with ft.VarDef([("x", (100,), "float32", "inout", "cpu"),
                    ("w", (100, 100), "float32", "input", "cpu")]) as (x, w):
        with ft.For("p", 0, 1000, label="Lp") as p:
            with ft.VarDef("y", (100,), "float32", "cache", "cpu") as y:
                with ft.For("i", 0, 100, label="Li0") as i:
                    y[i] = 0
                    with ft.For("j", 0, 100, label="Lj") as j:
                        y[i] += x[j] * w[i, j]
                with ft.For("i", 0, 100, label="Li1") as i:
                    x[i] = y[i]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(
        logs, ["parallelize(Li0, openmp, *)", "parallelize(Li1, openmp, *)"])
