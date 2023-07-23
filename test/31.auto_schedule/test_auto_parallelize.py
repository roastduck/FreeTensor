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


def test_3_levels():
    with ft.VarDef([("x", (100, 100, 100), "int32", "input", "cpu"),
                    ("y", (100, 100, 100), "int32", "output", "cpu")]) as (x,
                                                                           y):
        with ft.For("i", 0, 100, label="Li") as i:
            with ft.For("j", 0, 100, label="Lj") as j:
                with ft.For("k", 0, 100, label="Lk") as k:
                    y[i, j, k] = x[i, j, k] + 1

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.CPU())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(logs, [
        'merge(Li, Lj)', 'merge($merge{Li, Lj}, Lk)',
        'parallelize($merge{$merge{Li, Lj}, Lk}, openmp, *)'
    ])


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
        f'split(Lj, -1, {num_sm // 10}, 0)', 'merge(Li, $split.0{Lj})',
        'parallelize($merge{Li, $split.0{Lj}}, blockIdx.x, *)',
        'parallelize($split.1{Lj}, threadIdx.y, *)'
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
        'split(Lj, 256, -1, 0)', 'merge(Li, $split.0{Lj})',
        'parallelize($merge{Li, $split.0{Lj}}, blockIdx.x, *)',
        'parallelize($split.1{Lj}, threadIdx.y, *)'
    ])


# FIXME: Fix this test. Correctly prioritize all 3 scopes: `blockIdx.y`,
# `blockIdx.x` and `threadIdx.y` for dynamic loops.
#
# The problems lies on splitting. If we split an `n`-lengthed loop with `limits
# = {-1, 10, 10}` and `priority = {0, 2, 1}`, we first run `split(factor=100)`,
# resulting a seemingly 100-lengthed loop. But the loop is probably not full. If
# we treat it as 100, and continue `split(factor=10)`, the third scope will likely
# be full as 10. In this case the third scope will be more prioritized than the
# second scope, which is not what we want.
#
# To make it correct, we need to treat the seemingly 100-length loop with its real
# length `min(n, 100)`. However, this requires two improvements: 1) Changes to
# `split`, and more importantly 2) More accurate dependence analysis supporting
# dynamic divisor. A possible workaround is to analyze the dependence of a loop
# before we split it, and mark it with `no_deps`.
@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_basic_dynamic():
    with ft.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ft.VarDef([("x", (n[()], 10, 2), "int32", "input", "cpu"),
                        ("y", (n[()], 10, 2), "int32", "output", "cpu")
                       ]) as (x, y):
            with ft.For("i", 0, n[()], label="Li") as i:
                with ft.For("j", 0, 10, label="Lj") as j:
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
        'split(Li, %d, -1, 0)' %
        (num_sm * 256 // 10), 'parallelize($split.0{Li}, blockIdx.y, *)',
        'split($split.1{Li}, -1, %d, 0)' % num_sm,
        'parallelize($split.0{$split.1{Li}}, blockIdx.x, *)',
        'merge($split.1{$split.1{Li}}, Lj)',
        'parallelize($merge{$split.1{$split.1{Li}}, Lj}, threadIdx.y, *)'
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
def test_reduction_better_parallelized_1():
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


@pytest.mark.skipif(
    ft.CPU().target().n_cores() < 16 or ft.CPU().target().n_cores() > 1000,
    reason="This test is designed for systems with typical number of cores")
def test_reduction_better_parallelized_2():
    with ft.VarDef([("x", (2000,), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[...] = 0
        with ft.For("i", 0, 2000, label="Li") as i:
            y[...] += x[i]

    print(f"There are {ft.CPU().target().n_cores()} cores")
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.CPU())
    logs = list(map(str, s.logs()))
    print(logs)
    n_cores = ft.CPU().target().n_cores()
    assert fnmatch_list(logs, [
        f'split(Li, -1, {n_cores}, 0)',
        'parallelize($split.0{Li}, openmp, true)'
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
        "reorder($split.1{Lk}, $split.0{Lk}, *)", "split(Li, 8, -1, 0)",
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

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.GPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(logs[:3], [
        "split(Lk, 32, -1, 0)", "parallelize($split.1{Lk}, threadIdx.x, *)",
        "reorder($split.1{Lk}, $split.0{Lk}, *)"
    ])


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_gpu_warp_with_mod():
    with ft.VarDef([("x", (1000, 800), "int32", "input", "gpu/global"),
                    ("y", (1000, 2), "int32", "output", "gpu/global")]) as (x,
                                                                            y):
        with ft.For("i", 0, 1000, label="Li") as i:
            y[i, 0] = 0
            with ft.For("k", 0, 1000, label="Lk") as k:
                y[i, 0] += x[i, k % 800]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_parallelize(ft.GPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(logs, [
        "split(Lk, 32, -1, 0)", "parallelize($split.1{Lk}, threadIdx.x, *)",
        "reorder($split.1{Lk}, $split.0{Lk}, *)", "split(Li, 8, -1, 0)",
        "parallelize($split.0{Li}, blockIdx.x, *)",
        "parallelize($split.1{Li}, threadIdx.y, *)"
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
