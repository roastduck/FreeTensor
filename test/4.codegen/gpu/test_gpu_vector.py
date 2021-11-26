import ir
import ir.debug
import pytest
import numpy as np

target = ir.GPU()
device = ir.Device(target)


def test_vectorize():
    with ir.VarDef([
        ("x", (4, 64), "int32", "input", "gpu/global"),
        ("y", (4, 64), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 64, nid="L2") as j:
                y[i, j] = x[i, j] * 2
    func = ir.Func("main", ["x", "y"], [], ir.pop_ast())

    s = ir.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    assert "int4" in code

    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy().reshape(4, 64)

    y_std = x_np * 2
    assert np.array_equal(y_np, y_std)


def test_vectorize_with_non_vector_access():
    with ir.VarDef([
        ("x", (4,), "int32", "input", "gpu/global"),
        ("y", (4, 64), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 64, nid="L2") as j:
                y[i, j] = x[i] * 2
    func = ir.Func("main", ["x", "y"], [], ir.pop_ast())

    s = ir.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    assert "int4" in code

    x_np = np.random.randint(0, 100, (4,)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy().reshape(4, 64)

    y_std = np.broadcast_to(x_np * 2, (64, 4)).transpose()
    assert np.array_equal(y_np, y_std)


def test_vectorize_use_iter():
    with ir.VarDef("y", (4, 64), "int32", "output", "gpu/global") as y:
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 64, nid="L2") as j:
                y[i, j] = i + j
    func = ir.Func("main", ["y"], [], ir.pop_ast())

    s = ir.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    assert "int4" in code

    y_np = np.zeros((4, 64), dtype="int32")
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)(y=y_arr)
    y_np = y_arr.numpy().reshape(4, 64)

    y_std = np.array([[i + j for j in range(64)] for i in range(4)])
    assert np.array_equal(y_np, y_std)


def test_vectorize_fallback_to_shorter_when_not_divisible():
    with ir.VarDef([
        ("x", (4, 62), "int32", "input", "gpu/global"),
        ("y", (4, 62), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 62, nid="L2") as j:
                y[i, j] = x[i, j] * 2
    func = ir.Func("main", ["x", "y"], [], ir.pop_ast())

    s = ir.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    assert "int2" in code

    x_np = np.random.randint(0, 100, (4, 62)).astype("int32")
    y_np = np.zeros((4, 62), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy().reshape(4, 62)

    y_std = x_np * 2
    assert np.array_equal(y_np, y_std)


def test_vectorize_fallback_to_shorter_when_not_aligned():
    with ir.VarDef([
        ("x", (4, 66), "int32", "input", "gpu/global"),
        ("y", (4, 64), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 64, nid="L2") as j:
                y[i, j] = x[i, j + 2] * 2
    func = ir.Func("main", ["x", "y"], [], ir.pop_ast())

    s = ir.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    assert "int2" in code

    x_np = np.random.randint(0, 100, (4, 66)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy().reshape(4, 64)

    y_std = x_np[:, 2:] * 2
    assert np.array_equal(y_np, y_std)
