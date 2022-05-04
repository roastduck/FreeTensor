import freetensor as ft
import pytest
import numpy as np

target = ft.CPU()
device = ft.Device(target)


def test_omp_for():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4,), "int32", "input", "cpu"]
        y: ft.Var[(4,), "int32", "output", "cpu"]
        "nid: L1"
        for i in range(0, 4):
            y[i] = x[i] + 1

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            y[i] = x[i] + 1
    assert ft.pop_ast().match(test.body)

    s = ft.Schedule(test)
    s.parallelize("L1", "openmp")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ft.Array(x_np, ft.Device(target))
    y_arr = ft.Array(y_np, ft.Device(target))
    ft.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_omp_for_2():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 4), "int32", "input", "cpu"]
        y: ft.Var[(4, 4), "int32", "output", "cpu"]
        "nid: L1"
        for i in range(4):
            "nid: L2"
            for j in range(4):
                y[i, j] = x[i, j] + 1

    s = ft.Schedule(test)
    L12 = s.merge("L1", "L2")
    s.parallelize(L12, "openmp")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    x_np = np.array([[i + j for j in range(4)] for i in range(4)],
                    dtype="int32")
    y_np = np.zeros((4, 4), dtype="int32")
    x_arr = ft.Array(x_np, ft.Device(target))
    y_arr = ft.Array(y_np, ft.Device(target))
    ft.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([[i + j + 1 for j in range(4)] for i in range(4)],
                     dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_omp_for_collapse_nested():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: L1"
        for i in range(0, 4):
            "nid: L2"
            for j in range(0, 4):
                y[i, j] = x[i, j] + 1

    s = ft.Schedule(test)
    s.parallelize("L1", "openmp")
    s.parallelize("L2", "openmp")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "collapse(2)" in code
    x_np = np.random.rand(4, 4).astype("float32")
    y_np = np.zeros((4, 4), dtype="float32")
    x_arr = ft.Array(x_np, ft.Device(target))
    y_arr = ft.Array(y_np, ft.Device(target))
    ft.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    assert np.all(np.isclose(y_np, x_np + 1))


def test_parallelize_parametric_access_1():

    @ft.transform
    def test(idx, y):
        idx: ft.Var[(10,), "int32", "input", "cpu"]
        y: ft.Var[(100,), "int32", "inout", "cpu"]
        "nid: L1"
        for i in range(10):
            "nid: L2"
            for j in range(10):
                y[idx[i] + j] += j

    s = ft.Schedule(test)
    s.parallelize("L1", "openmp")
    func = ft.lower(s.func(), target)
    print(func)

    # idx[i] + j for different i may be the same, so we need atomic
    code = ft.codegen(func, target)
    print(code)
    assert "#pragma omp atomic" in code
    assert "+=" in code


def test_parallelize_parametric_access_2():

    @ft.transform
    def test(idx, y):
        idx: ft.Var[(10,), "int32", "input", "cpu"]
        y: ft.Var[(100,), "int32", "inout", "cpu"]
        "nid: L1"
        for i in range(10):
            "nid: L2"
            for j in range(10):
                y[idx[i] + j] += j

    s = ft.Schedule(test)
    s.parallelize("L2", "openmp")
    func = ft.lower(s.func(), target)
    print(func)

    # idx[i] + j for the same i but different j must be different, so we do not need atomic
    code = ft.codegen(func, target)
    print(code)
    assert "#pragma omp atomic" not in code
    assert "+=" in code


def test_unroll_for():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4,), "int32", "input", "cpu"]
        y: ft.Var[(4,), "int32", "output", "cpu"]
        "nid: L1"
        for i in range(0, 4):
            y[i] = x[i] + 1

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            y[i] = x[i] + 1
    assert ft.pop_ast().match(test.body)

    s = ft.Schedule(test)
    s.unroll("L1")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "#pragma GCC unroll" in code
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_vectorize_for():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4,), "int32", "input", "cpu"]
        y: ft.Var[(4,), "int32", "output", "cpu"]
        "nid: L1"
        for i in range(0, 4):
            y[i] = x[i] + 1

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            y[i] = x[i] + 1
    assert ft.pop_ast().match(test.body)

    s = ft.Schedule(test)
    s.vectorize("L1")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "#pragma omp simd" in code
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)
