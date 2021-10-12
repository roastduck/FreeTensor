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


def test_gpu_basic():
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
    print(s.logs())
    assert s.logs() == [
        "merge(Li, Lj)", "split(merged.Li.Lj, factor=-1, nparts=80)",
        "split(merged.Li.Lj.1, factor=1024, nparts=-1)",
        "parallelize(merged.Li.Lj.0, blockIdx.y)",
        "parallelize(merged.Li.Lj.1.0, blockIdx.x)",
        "parallelize(merged.Li.Lj.1.1, threadIdx.x)"
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


def test_gpu_warp():
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
        "split(Li, factor=-1, nparts=80)", "split(Li.1, factor=32, nparts=-1)",
        "parallelize(Li.0, blockIdx.y)", "parallelize(Li.1.1, threadIdx.y)"
    ]
