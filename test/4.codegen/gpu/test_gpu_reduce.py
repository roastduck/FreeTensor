import ir
import ir.debug
import pytest
import numpy as np

target = ir.GPU()
device = ir.Device(target)


def test_parallel_reduction():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 64), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            "nid: L2"
            for j in range(0, 64):
                y[i] = y[i] + x[i, j]

    with ir.VarDef([
        ("x", (4, 64), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 64, nid="L2") as j:
                y[i] = y[i] + x[i, j]
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L1", "blockIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    assert "atomicAdd" not in code
    print(ir.debug.with_line_no(code))
    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.sum(x_np, axis=1)
    assert np.array_equal(y_np, y_std)


def test_parallel_reduction_on_2_vars():

    @ir.transform
    def test(x, y, z):
        ir.declare_var(x, (4, 64), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        ir.declare_var(z, (4,), "int32", "output", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            "nid: L2"
            for j in range(0, 64):
                y[i] = y[i] + x[i, j]
                z[i] = z[i] + x[i, j] * 2

    s = ir.Schedule(test)
    s.parallelize("L1", "blockIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    assert "atomicAdd" not in code
    print(ir.debug.with_line_no(code))
    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4,), dtype="int32")
    z_np = np.zeros((4,), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    z_arr = ir.Array(z_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr, z=z_arr)
    y_np = y_arr.numpy()
    z_np = z_arr.numpy()

    y_std = np.sum(x_np, axis=1)
    z_std = np.sum(x_np, axis=1) * 2
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(z_np, z_std)


def test_parallel_reduction_on_array():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 64, 64), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 64), "int32", "output", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            "nid: L2"
            for j in range(0, 64):
                "nid: L3"
                for k in range(0, 64):
                    y[i, k] = y[i, k] + x[i, j, k]

    s = ir.Schedule(test)
    s.parallelize("L1", "blockIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    assert "atomicAdd" not in code
    print(ir.debug.with_line_no(code))
    x_np = np.random.randint(0, 100, (4, 64, 64)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy().reshape(4, 64)

    y_std = np.sum(x_np, axis=1)
    assert np.array_equal(y_np, y_std)


def test_atomic_reduction():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 64), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 2), "int32", "output", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            "nid: L2"
            for j in range(0, 64):
                y[i, j % 2] += x[i, j]

    with ir.VarDef([
        ("x", (4, 64), "int32", "input", "gpu/global"),
        ("y", (4, 2), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 64, nid="L2") as j:
                y[i, j % 2] += x[i, j]
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L1", "blockIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    assert "atomicAdd" in code
    assert "+=" not in code
    print(ir.debug.with_line_no(code))
    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4, 2), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy().reshape(4, 2)

    y_std = np.sum(x_np.reshape((4, 32, 2)), axis=1)
    assert np.array_equal(y_np, y_std)


def test_atomic_reduction_2_stmts_on_1_var_across_blocks():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 64), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 64), "int32", "inout", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            "nid: L2"
            for j in range(0, 64):
                y[i, j] += x[i, j]
                if j > 0:
                    y[i, j - 1] += x[i, j]

    s = ir.Schedule(test)
    s.parallelize("L2", "blockIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    print(code)
    assert code.count("atomicAdd") == 2
    assert "+=" not in code
    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy().reshape(4, 64)

    y_std = x_np
    y_std[:, :-1] += x_np[:, 1:]
    assert np.array_equal(y_np, y_std)


def test_no_atomic_reduction_2_stmts_on_1_var_across_threads():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 64), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 64), "int32", "inout", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            "nid: L2"
            for j in range(0, 64):
                y[i, j] += x[i, j]
                if j > 0:
                    y[i, j - 1] += x[i, j]

    s = ir.Schedule(test)
    s.parallelize("L2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    print(code)
    assert "atomicAdd" not in code
    assert "__syncthreads" in code
    assert "+=" in code
    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy().reshape(4, 64)

    y_std = x_np
    y_std[:, :-1] += x_np[:, 1:]
    assert np.array_equal(y_np, y_std)


def test_serial_reduction():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 64), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            "nid: L2"
            for j in range(0, 64):
                y[i] = y[i] + x[i, j]

    with ir.VarDef([
        ("x", (4, 64), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 64, nid="L2") as j:
                y[i] = y[i] + x[i, j]
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L1", "blockIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    assert "atomicAdd" not in code
    assert "+=" in code
    print(ir.debug.with_line_no(code))
    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.sum(x_np, axis=1)
    assert np.array_equal(y_np, y_std)
