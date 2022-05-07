import freetensor as ft
from freetensor import debug
import pytest
import numpy as np

if not ft.with_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

target = ft.GPU()
device = ft.Device(target)


def test_vectorize():
    with ft.VarDef([
        ("x", (4, 64), "int32", "input", "gpu/global"),
        ("y", (4, 64), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 64, nid="L2") as j:
                y[i, j] = x[i, j] * 2
    func = ft.Func("main", ["x", "y"], [], ft.pop_ast())

    s = ft.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ft.lower(s.func(), target, verbose=1)

    code = ft.codegen(func, target, verbose=True)
    assert "int4" in str(code)

    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ft.Array(x_np, device)
    y_arr = ft.Array(y_np, device)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = x_np * 2
    assert np.array_equal(y_np, y_std)


def test_vectorize_with_non_vector_access():
    with ft.VarDef([
        ("x", (4,), "int32", "input", "gpu/global"),
        ("y", (4, 64), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 64, nid="L2") as j:
                y[i, j] = x[i] * 2
    func = ft.Func("main", ["x", "y"], [], ft.pop_ast())

    s = ft.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ft.lower(s.func(), target, verbose=1)

    code = ft.codegen(func, target, verbose=True)
    assert "int4" in str(code)

    x_np = np.random.randint(0, 100, (4,)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ft.Array(x_np, device)
    y_arr = ft.Array(y_np, device)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.broadcast_to(x_np * 2, (64, 4)).transpose()
    assert np.array_equal(y_np, y_std)


def test_vectorize_use_iter():
    with ft.VarDef("y", (4, 64), "int32", "output", "gpu/global") as y:
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 64, nid="L2") as j:
                y[i, j] = i + j
    func = ft.Func("main", ["y"], [], ft.pop_ast())

    s = ft.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ft.lower(s.func(), target, verbose=1)

    code = ft.codegen(func, target, verbose=True)
    assert "int4" in str(code)

    y_np = np.zeros((4, 64), dtype="int32")
    y_arr = ft.Array(y_np, device)
    driver = ft.build_binary(code, device)(y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([[i + j for j in range(64)] for i in range(4)])
    assert np.array_equal(y_np, y_std)


def test_vectorize_fallback_to_shorter_when_not_divisible():
    with ft.VarDef([
        ("x", (4, 62), "int32", "input", "gpu/global"),
        ("y", (4, 62), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 62, nid="L2") as j:
                y[i, j] = x[i, j] * 2
    func = ft.Func("main", ["x", "y"], [], ft.pop_ast())

    s = ft.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ft.lower(s.func(), target, verbose=1)

    code = ft.codegen(func, target, verbose=True)
    assert "int2" in str(code)

    x_np = np.random.randint(0, 100, (4, 62)).astype("int32")
    y_np = np.zeros((4, 62), dtype="int32")
    x_arr = ft.Array(x_np, device)
    y_arr = ft.Array(y_np, device)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = x_np * 2
    assert np.array_equal(y_np, y_std)


def test_vectorize_fallback_to_shorter_when_not_aligned():
    with ft.VarDef([
        ("x", (4, 66), "int32", "input", "gpu/global"),
        ("y", (4, 64), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 64, nid="L2") as j:
                y[i, j] = x[i, j + 2] * 2
    func = ft.Func("main", ["x", "y"], [], ft.pop_ast())

    s = ft.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ft.lower(s.func(), target, verbose=1)

    code = ft.codegen(func, target, verbose=True)
    assert "int2" in str(code)

    x_np = np.random.randint(0, 100, (4, 66)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ft.Array(x_np, device)
    y_arr = ft.Array(y_np, device)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = x_np[:, 2:] * 2
    assert np.array_equal(y_np, y_std)
