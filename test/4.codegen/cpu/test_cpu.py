import ir
import pytest
import numpy as np

target = ir.CPU()
device = ir.Device(target)


def test_omp_for():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4,), "int32", "input", "cpu")
        ir.declare_var(y, (4,), "int32", "output", "cpu")
        "nid: L1"
        for i in range(0, 4):
            y[i] = x[i] + 1

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            y[i] = x[i] + 1
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L1", "openmp")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(target))
    y_arr = ir.Array(y_np, ir.Device(target))
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_omp_for_2():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 4), "int32", "input", "cpu")
        ir.declare_var(y, (4, 4), "int32", "output", "cpu")
        "nid: L1"
        for i in range(4):
            "nid: L2"
            for j in range(4):
                y[i, j] = x[i, j] + 1

    s = ir.Schedule(test)
    L12 = s.merge("L1", "L2")
    s.parallelize(L12, "openmp")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    x_np = np.array([[i + j for j in range(4)] for i in range(4)],
                    dtype="int32")
    y_np = np.zeros((4, 4), dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(target))
    y_arr = ir.Array(y_np, ir.Device(target))
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy().reshape(4, 4)

    y_std = np.array([[i + j + 1 for j in range(4)] for i in range(4)],
                     dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_parallelize_parametric_access_1():

    @ir.transform
    def test(idx, y):
        ir.declare_var(idx, (10,), "int32", "input", "cpu")
        ir.declare_var(y, (100,), "int32", "inout", "cpu")
        "nid: L1"
        for i in range(10):
            "nid: L2"
            for j in range(10):
                y[idx[i] + j] += j

    s = ir.Schedule(test)
    s.parallelize("L1", "openmp")
    func = ir.lower(s.func(), target)
    print(func)

    # idx[i] + j for different i may be the same, so we need atomic
    code = ir.codegen(func, target)
    print(code)
    assert "#pragma omp atomic" in code
    assert "+=" in code


def test_parallelize_parametric_access_2():

    @ir.transform
    def test(idx, y):
        ir.declare_var(idx, (10,), "int32", "input", "cpu")
        ir.declare_var(y, (100,), "int32", "inout", "cpu")
        "nid: L1"
        for i in range(10):
            "nid: L2"
            for j in range(10):
                y[idx[i] + j] += j

    s = ir.Schedule(test)
    s.parallelize("L2", "openmp")
    func = ir.lower(s.func(), target)
    print(func)

    # idx[i] + j for the same i but different j must be different, so we do not need atomic
    code = ir.codegen(func, target)
    print(code)
    assert "#pragma omp atomic" not in code
    assert "+=" in code


def test_unroll_for():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4,), "int32", "input", "cpu")
        ir.declare_var(y, (4,), "int32", "output", "cpu")
        "nid: L1"
        for i in range(0, 4):
            y[i] = x[i] + 1

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            y[i] = x[i] + 1
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.unroll("L1")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    assert "#pragma GCC unroll" in code
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_vectorize_for():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4,), "int32", "input", "cpu")
        ir.declare_var(y, (4,), "int32", "output", "cpu")
        "nid: L1"
        for i in range(0, 4):
            y[i] = x[i] + 1

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            y[i] = x[i] + 1
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.vectorize("L1")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    assert "#pragma omp simd" in code
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)
