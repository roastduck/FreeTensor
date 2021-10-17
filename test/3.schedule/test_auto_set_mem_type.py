import ir
import pytest


def test_gpu_basic():
    with ir.VarDef([("x", (1000, 1000), "int32", "input", "gpu/global"),
                    ("y", (1000, 1000), "int32", "output", "gpu/global")
                   ]) as (x, y):
        with ir.For("i", 0, 1000, nid="Li") as i:
            ir.MarkNid("V_t")
            with ir.VarDef("t", (), "int32", "cache", "gpu/global") as t:
                t[()] = 0
                with ir.For("j", 0, 1000, nid="Lj1") as j:
                    t[()] += x[i, j]
                with ir.For("j", 0, 1000, nid="Lj2") as j:
                    y[i, j] = x[i, j] - t[()]

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.parallelize('Li', 'blockIdx.x')
    s.parallelize('Lj2', 'threadIdx.x')
    s.auto_set_mem_type(ir.GPU())
    print(s.ast())
    print(s.logs())
    assert s.logs()[2:] == ["set_mem_type(V_t, [GPUShared])"]


def test_gpu_local_across_loops():
    with ir.VarDef([("x", (1000, 1000), "int32", "input", "gpu/global"),
                    ("y", (1000, 1000), "int32", "output", "gpu/global")
                   ]) as (x, y):
        with ir.For("i", 0, 1000, nid="Li") as i:
            ir.MarkNid("V_t")
            with ir.VarDef("t", (1000,), "int32", "cache", "gpu/global") as t:
                with ir.For("j", 0, 1000, nid="Lj1") as j:
                    t[j] = x[i, j] + 1
                with ir.For("j", 0, 1000, nid="Lj2") as j:
                    y[i, j] = t[j] * 2

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.parallelize('Li', 'blockIdx.x')
    s.parallelize('Lj1', 'threadIdx.x')
    s.parallelize('Lj2', 'threadIdx.x')
    s.auto_set_mem_type(ir.GPU())
    print(s.ast())
    print(s.logs())
    assert s.logs()[3:] == ["set_mem_type(V_t, [GPULocal])"]
