import ir
import ir.debug
import pytest
import numpy as np

target = ir.GPU()
device = ir.Device(target)


def test_warpshuffle_reverse():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 32), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 32), "int32", "output", "gpu/global")
        'nid: L0'
        for i in range(0, 4):
            value = ir.create_var((32,), "int32", "gpu/warp")
            'nid: L1'
            for j in range(0, 32):
                value[j] = x[i, j]
            'nid: L2'
            for j in range(0, 32):
                y[i, j] = value[31 - j]

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (4, 32), "int32", "input", "gpu/global"),
        ("y", (4, 32), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For(".blockIdx.x", 0, 4) as i:
            with ir.VarDef("value", (32,), "int32", "cache", "gpu/warp") as value:
                with ir.For(".threadIdx.x", 0, 32) as j:
                    value[j] = x[i, j]
                with ir.For(".threadIdx.x", 0, 32) as j:
                    y[i, j] = value[31 - j]
    assert ir.pop_ast().match(test.body)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    x_np = np.array([list(range(num * 32, (num + 1) * 32)) for num in range(4)], dtype="int32")
    y_np = np.zeros((4, 32), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    
    ir.Driver(func, code, device)(x=x_arr,y=y_arr)

    y_np = y_arr.numpy();
    y_std = np.array([list(range(num * 32 + 31, num * 32 - 1, -1)) for num in range(4)], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_warpshuffle_sum():

    @ir.transform
    def test(x, y, z):
        ir.declare_var(x, (4, 32), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 32), "int32", "input", "gpu/global")
        ir.declare_var(z, (4, 32), "int32", "output", "gpu/global")
        'nid: L0'
        for i in range(0, 4):
            value1 = ir.create_var((32,), "int32", "gpu/warp")
            value2 = ir.create_var((32,), "int32", "gpu/warp")
            'nid: L1'
            for j in range(0, 32):
                value1[j] = x[i, j]
                value2[j] = y[i, j]
            'nid: L2'
            for j in range(0, 32):
                value2[j] += value1[31 - j]
            'nid: L3'
            for j in range(0, 32):
                z[i, j] = value2[j]

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    s.parallelize("L3", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (4, 32), "int32", "input", "gpu/global"),
        ("y", (4, 32), "int32", "input", "gpu/global"),
        ("z", (4, 32), "int32", "output", "gpu/global"),
    ]) as (x, y, z):
        with ir.For(".blockIdx.x", 0, 4) as i:
            with ir.VarDef("value1", (32,), "int32", "cache", "gpu/warp") as value1:
                with ir.VarDef("value2", (32,), "int32", "cache", "gpu/warp") as value2:
                    with ir.For(".threadIdx.x", 0, 32) as j:
                        value1[j] = x[i, j]
                        value2[j] = y[i, j]
                    with ir.For(".threadIdx.x", 0, 32) as j:
                        value2[j] += value1[31 - j]
                    with ir.For(".threadIdx.x", 0, 32) as j:
                        z[i, j] = value2[j]
    assert ir.pop_ast().match(test.body)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    x_np = np.array([list(range(num * 32, (num + 1) * 32)) for num in range(4)], dtype="int32")
    y_np = np.array([list(range(num * 32, (num + 1) * 32)) for num in range(4)], dtype="int32")
    z_np = np.zeros((4, 32), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    z_arr = ir.Array(z_np, device)
    
    ir.Driver(func, code, device)(x=x_arr,y=y_arr,z=z_arr)

    z_np = z_arr.numpy();
    z_std = np.array([[num * 64 + 31 for k in range(32)] for num in range(4)], dtype="int32")
    assert np.array_equal(z_np, z_std)