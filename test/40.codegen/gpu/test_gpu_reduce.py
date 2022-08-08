import freetensor as ft
from freetensor import debug
import pytest
import numpy as np

if not ft.with_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

device = ft.GPU()
target = device.target()


def test_parallel_reduction():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 64), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        #! nid: L1
        for i in range(0, 4):
            #! nid: L2
            for j in range(0, 64):
                y[i] = y[i] + x[i, j]

    with ft.VarDef([
        ("x", (4, 64), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 64, nid="L2") as j:
                y[i] = y[i] + x[i, j]
    assert ft.pop_ast().match(test.body)

    s = ft.Schedule(test)
    s.parallelize("L1", "blockIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    code = ft.codegen(func, target)
    assert "atomicAdd" not in str(code)
    print(debug.with_line_no(code))
    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.sum(x_np, axis=1)
    assert np.array_equal(y_np, y_std)


def test_parallel_reduction_on_2_vars():

    @ft.transform
    def test(x, y, z):
        x: ft.Var[(4, 64), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        z: ft.Var[(4,), "int32", "output", "gpu/global"]
        #! nid: L1
        for i in range(0, 4):
            #! nid: L2
            for j in range(0, 64):
                y[i] = y[i] + x[i, j]
                z[i] = z[i] + x[i, j] * 2

    s = ft.Schedule(test)
    s.parallelize("L1", "blockIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    code = ft.codegen(func, target)
    assert "atomicAdd" not in str(code)
    print(debug.with_line_no(code))
    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4,), dtype="int32")
    z_np = np.zeros((4,), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    z_arr = ft.Array(z_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr, z=z_arr)
    y_np = y_arr.numpy()
    z_np = z_arr.numpy()

    y_std = np.sum(x_np, axis=1)
    z_std = np.sum(x_np, axis=1) * 2
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(z_np, z_std)


def test_parallel_reduction_on_array():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 64, 64), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 64), "int32", "output", "gpu/global"]
        #! nid: L1
        for i in range(0, 4):
            #! nid: L2
            for j in range(0, 64):
                #! nid: L3
                for k in range(0, 64):
                    y[i, k] = y[i, k] + x[i, j, k]

    s = ft.Schedule(test)
    s.parallelize("L1", "blockIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    code = ft.codegen(func, target)
    assert "atomicAdd" not in str(code)
    print(debug.with_line_no(code))
    x_np = np.random.randint(0, 100, (4, 64, 64)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.sum(x_np, axis=1)
    assert np.array_equal(y_np, y_std)


def test_atomic_reduction():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 64), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 2), "int32", "output", "gpu/global"]
        #! nid: L1
        for i in range(0, 4):
            #! nid: L2
            for j in range(0, 64):
                y[i, j % 2] += x[i, j]

    with ft.VarDef([
        ("x", (4, 64), "int32", "input", "gpu/global"),
        ("y", (4, 2), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 64, nid="L2") as j:
                y[i, j % 2] += x[i, j]
    assert ft.pop_ast().match(test.body)

    s = ft.Schedule(test)
    s.parallelize("L1", "blockIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    code = ft.codegen(func, target)
    assert "atomicAdd" in str(code)
    assert "+=" not in str(code)
    print(debug.with_line_no(code))
    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4, 2), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.sum(x_np.reshape((4, 32, 2)), axis=1)
    assert np.array_equal(y_np, y_std)


def test_atomic_reduction_2_stmts_on_1_var_across_blocks():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 64), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 64), "int32", "inout", "gpu/global"]
        #! nid: L1
        for i in range(0, 4):
            #! nid: L2
            for j in range(0, 64):
                y[i, j] += x[i, j]
                if j > 0:
                    y[i, j - 1] += x[i, j]

    s = ft.Schedule(test)
    s.parallelize("L2", "blockIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    code = ft.codegen(func, target, verbose=True)
    assert str(code).count("atomicAdd") == 2
    assert "+=" not in str(code)
    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = x_np
    y_std[:, :-1] += x_np[:, 1:]
    assert np.array_equal(y_np, y_std)


def test_no_atomic_reduction_2_stmts_on_1_var_across_threads():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 64), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 64), "int32", "inout", "gpu/global"]
        #! nid: L1
        for i in range(0, 4):
            #! nid: L2
            for j in range(0, 64):
                y[i, j] += x[i, j]
                if j > 0:
                    y[i, j - 1] += x[i, j]

    s = ft.Schedule(test)
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    code = ft.codegen(func, target, verbose=True)
    assert "atomicAdd" not in str(code)
    assert "__syncthreads" in str(code)
    assert "+=" in str(code)
    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = x_np
    y_std[:, :-1] += x_np[:, 1:]
    assert np.array_equal(y_np, y_std)


def test_serial_reduction():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 64), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        #! nid: L1
        for i in range(0, 4):
            #! nid: L2
            for j in range(0, 64):
                y[i] = y[i] + x[i, j]

    with ft.VarDef([
        ("x", (4, 64), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 64, nid="L2") as j:
                y[i] = y[i] + x[i, j]
    assert ft.pop_ast().match(test.body)

    s = ft.Schedule(test)
    s.parallelize("L1", "blockIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    code = ft.codegen(func, target)
    assert "atomicAdd" not in str(code)
    assert "+=" in str(code)
    print(debug.with_line_no(code))
    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.sum(x_np, axis=1)
    assert np.array_equal(y_np, y_std)
