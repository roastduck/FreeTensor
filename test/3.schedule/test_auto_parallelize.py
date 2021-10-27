import ir
import pytest


def test_cpu_basic():
    with ir.VarDef([("x", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000, 1000), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 1000, nid="Li") as i:
            with ir.For("j", 0, 1000, nid="Lj") as j:
                y[i, j] = x[i, j] + 1

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.auto_parallelize(ir.CPU())
    print(s.logs())
    assert s.logs() == ["merge(Li, Lj)", "parallelize(merged.Li.Lj, openmp)"]


def test_gpu_basic_static_small():
    with ir.VarDef([("x", (10, 10, 2), "int32", "input", "cpu"),
                    ("y", (10, 10, 2), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 10, nid="Li") as i:
            with ir.For("j", 0, 10, nid="Lj") as j:
                y[i, j, 0] = x[i, j, 0] + 1

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.auto_parallelize(ir.GPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == [
        "merge(Li, Lj)", "split(merged.Li.Lj, factor=-1, nparts=80)",
        "parallelize(merged.Li.Lj.0, blockIdx.x)",
        "parallelize(merged.Li.Lj.1, threadIdx.x)"
    ]


def test_gpu_basic_static_large():
    with ir.VarDef([("x", (1000, 1000, 2), "int32", "input", "cpu"),
                    ("y", (1000, 1000, 2), "int32", "output", "cpu")]) as (x,
                                                                           y):
        with ir.For("i", 0, 1000, nid="Li") as i:
            with ir.For("j", 0, 1000, nid="Lj") as j:
                y[i, j, 0] = x[i, j, 0] + 1

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.auto_parallelize(ir.GPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == [
        "merge(Li, Lj)", "split(merged.Li.Lj, factor=256, nparts=-1)",
        "parallelize(merged.Li.Lj.0, blockIdx.x)",
        "parallelize(merged.Li.Lj.1, threadIdx.x)"
    ]


def test_gpu_basic_dynamic():
    with ir.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ir.VarDef([("x", (n[()], 1000, 2), "int32", "input", "cpu"),
                        ("y", (n[()], 1000, 2), "int32", "output", "cpu")
                       ]) as (x, y):
            with ir.For("i", 0, n[()], nid="Li") as i:
                with ir.For("j", 0, 1000, nid="Lj") as j:
                    y[i, j, 0] = x[i, j, 0] + 1

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.auto_parallelize(ir.GPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == [
        "merge(Li, Lj)", "split(merged.Li.Lj, factor=80, nparts=-1)",
        "reorder(merged.Li.Lj.1, merged.Li.Lj.0)",
        "split(merged.Li.Lj.0, factor=256, nparts=-1)",
        "parallelize(merged.Li.Lj.1, blockIdx.y)",
        "parallelize(merged.Li.Lj.0.0, blockIdx.x)",
        "parallelize(merged.Li.Lj.0.1, threadIdx.x)"
    ]


def test_non_parallelizable():
    with ir.VarDef([("x", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 1000, nid="Li") as i:
            y[i] = 0
            with ir.For("j", 0, 1000, nid="Lj") as j:
                y[i] = y[i] * 2 + x[i, j]

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.auto_parallelize(ir.CPU())
    print(s.logs())
    assert s.logs() == ["parallelize(Li, openmp)"]


def test_reduction_better_not_parallelized():
    with ir.VarDef([("x", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 1000, nid="Li") as i:
            y[i] = 0
            with ir.For("j", 0, 1000, nid="Lj") as j:
                y[i] += x[i, j]

    ast = ir.make_reduction(ir.pop_ast())
    print(ast)
    s = ir.Schedule(ast)
    s.auto_parallelize(ir.CPU())
    print(s.logs())
    assert s.logs() == ["parallelize(Li, openmp)"]


def test_gpu_warp_static():
    with ir.VarDef([("x", (1000, 1000), "int32", "input", "gpu/global"),
                    ("y", (1000, 2), "int32", "output", "gpu/global")]) as (x,
                                                                            y):
        with ir.For("i", 0, 1000, nid="Li") as i:
            y[i, 0] = 0
            with ir.For("k", 0, 1000, nid="Lk") as k:
                y[i, 0] += x[i, k]

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.auto_parallelize(ir.GPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == [
        "split(Lk, factor=32, nparts=-1)", "parallelize(Lk.1, threadIdx.x)",
        "reorder(Lk.1, Lk.0)", "split(Li, factor=8, nparts=-1)",
        "parallelize(Li.0, blockIdx.x)", "parallelize(Li.1, threadIdx.y)"
    ]


def test_gpu_warp_dynamic():
    with ir.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ir.VarDef([("x", (n[()], n[()]), "int32", "input", "gpu/global"),
                        ("y", (n[()], 2), "int32", "output", "gpu/global")
                       ]) as (x, y):
            with ir.For("i", 0, n[()], nid="Li") as i:
                y[i, 0] = 0
                with ir.For("k", 0, n[()], nid="Lk") as k:
                    y[i, 0] += x[i, k]

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.auto_parallelize(ir.GPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == [
        "split(Lk, factor=32, nparts=-1)", "parallelize(Lk.1, threadIdx.x)",
        "reorder(Lk.1, Lk.0)", "split(Li, factor=80, nparts=-1)",
        "reorder(Li.1, Li.0)", "split(Li.0, factor=8, nparts=-1)",
        "parallelize(Li.1, blockIdx.y)", "parallelize(Li.0.0, blockIdx.x)",
        "parallelize(Li.0.1, threadIdx.y)"
    ]
