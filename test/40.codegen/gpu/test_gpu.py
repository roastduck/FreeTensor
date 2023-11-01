import freetensor as ft
from freetensor import debug
import pytest
import numpy as np

if not ft.with_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

device = ft.GPU()
target = device.target()


def test_basic():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4,), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        #! label: L1
        for i in range(0, 4):
            y[i] = x[i] + 1

    with ft.VarDef([
        ("x", (4,), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            y[i] = x[i] + 1
    assert ft.pop_ast().match(test.body)

    with device:
        s = ft.Schedule(test)
        s.parallelize("L1", "threadIdx.x")
        func = ft.lower(s.func(), verbose=1)
        code = ft.codegen(func, verbose=True)
        x_np = np.array([1, 2, 3, 4], dtype="int32")
        y_np = np.zeros((4,), dtype="int32")
        x_arr = ft.Array(x_np)
        y_arr = ft.Array(y_np)
        ft.build_binary(code)(x=x_arr, y=y_arr)
        y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_error_wrong_target():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4,), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        #! label: L1
        for i in range(0, 4):
            y[i] = x[i] + 1

    with pytest.raises(ft.DriverError):
        with device:
            s = ft.Schedule(test)
            s.parallelize("L1", "threadIdx.x")
            func = ft.lower(s.func(), verbose=1)
            code = ft.codegen(func, ft.CPU(), verbose=True)
            ft.build_binary(code)(x=x_arr, y=y_arr)


def test_define_output_inside_kernel():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4,), "int32", "input", "gpu/global"]
        #! label: L1
        for i in range(0, 4):
            y: ft.Var[(4,), "int32", "output", "gpu/global"]
            y[i] = x[i] + 1

    with ft.VarDef("x", (4,), "int32", "input", "gpu/global") as x:
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.VarDef("y", (4,), "int32", "output", "gpu/global") as y:
                y[i] = x[i] + 1
    assert ft.pop_ast().match(test.body)

    with device:
        s = ft.Schedule(test)
        s.parallelize("L1", "threadIdx.x")
        func = ft.lower(s.func(), verbose=1)
        code = ft.codegen(func, verbose=True)
        x_np = np.array([1, 2, 3, 4], dtype="int32")
        y_np = np.zeros((4,), dtype="int32")
        x_arr = ft.Array(x_np)
        y_arr = ft.Array(y_np)
        ft.build_binary(code)(x=x_arr, y=y_arr)
        y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_return_value_and_runtime_allocation():

    @ft.transform
    def test(x):
        x: ft.Var[(4, 4), "int32", "input", "gpu/global"]
        y = ft.empty((4, 4), "int32", "gpu/global")
        #! label: L1
        for i in range(4):
            #! label: L2
            for j in range(4):
                y[i, j] = x[i, j] * 2 + 1
        return y

    with device:
        s = ft.Schedule(test)
        s.parallelize("L1", "blockIdx.x")
        s.parallelize("L2", "threadIdx.x")
        func = ft.lower(s.func(), verbose=1)
        code = ft.codegen(func, verbose=True)
        x_np = np.random.randint(0, 100, (4, 4)).astype("int32")
        x_arr = ft.Array(x_np)
        y_arr = ft.build_binary(code)(x_arr)
        y_np = y_arr.numpy()

    assert np.array_equal(y_np, x_np * 2 + 1)


def test_scalar():

    @ft.transform
    def test(x):
        x: ft.Var[(4,), "int32", "input", "gpu/global"]
        y1 = ft.empty((), "int32", "gpu/global")
        y2 = ft.empty((), "int32", "gpu/global")
        y3 = ft.empty((), "int32", "gpu/global")
        y4 = ft.empty((), "int32", "gpu/global")
        #! label: L1
        for i in range(4):
            if i == 0:
                y1[...] = x[i]
            elif i == 1:
                y2[...] = x[i]
            elif i == 2:
                y3[...] = x[i]
            else:
                y4[...] = x[i]
        z = ft.empty((4,), "int32", "gpu/global")
        #! label: L2
        for i in range(4):
            if i == 0:
                z[i] = y1[...] + y3[...]
            elif i == 1:
                z[i] = y2[...] + y4[...]
            elif i == 2:
                z[i] = y1[...] + y2[...]
            else:
                z[i] = y3[...] + y4[...]
        return z

    with device:
        s = ft.Schedule(test)
        s.parallelize("L1", "blockIdx.x")
        s.parallelize("L2", "blockIdx.x")
        func = ft.lower(s.func(), verbose=1)
        code = ft.codegen(func, verbose=True)
        x_np = np.array([1, 2, 3, 4]).astype("int32")
        x_arr = ft.Array(x_np)
        z_arr = ft.build_binary(code)(x_arr)
        z_np = z_arr.numpy()

    assert np.array_equal(z_np, [4, 6, 3, 7])


def test_shmem_scalar():

    @ft.transform
    def test(x):
        x: ft.Var[(4,), "int32", "input", "gpu/global"]
        t1 = ft.empty((), "int32", "gpu/shared")
        t2 = ft.empty((), "int32", "gpu/shared")
        t3 = ft.empty((), "int32", "gpu/shared")
        t4 = ft.empty((), "int32", "gpu/shared")
        #! label: L1
        for i in range(4):
            if i == 0:
                t1[...] = x[i]
            elif i == 1:
                t2[...] = x[i]
            elif i == 2:
                t3[...] = x[i]
            else:
                t4[...] = x[i]
        y = ft.empty((4,), "int32", "gpu/global")
        #! label: L2
        for i in range(4):
            if i == 0:
                y[i] = t1[...] + t3[...]
            elif i == 1:
                y[i] = t2[...] + t4[...]
            elif i == 2:
                y[i] = t1[...] + t2[...]
            else:
                y[i] = t3[...] + t4[...]
        return y

    with device:
        s = ft.Schedule(test)
        s.parallelize("L1", "threadIdx.x")
        s.parallelize("L2", "threadIdx.x")
        func = ft.lower(s.func(), verbose=1)
        code = ft.codegen(func, verbose=True)
        x_np = np.array([1, 2, 3, 4]).astype("int32")
        x_arr = ft.Array(x_np)
        y_arr = ft.build_binary(code)(x_arr)
        y_np = y_arr.numpy()

    assert np.array_equal(y_np, [4, 6, 3, 7])


def test_split_by_block_and_bind():

    @ft.transform
    def test(x, y):
        x: ft.Var[(100,), "int32", "input", "gpu/global"]
        y: ft.Var[(100,), "int32", "output", "gpu/global"]
        #! label: L1
        for i in range(0, 100):
            y[i] = x[i] + 1

    s = ft.Schedule(test)
    outer, inner = s.split("L1", nparts=3)
    s.parallelize(outer, "blockIdx.x")
    s.parallelize(inner, "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    with ft.VarDef([
        ("x", (100,), "int32", "input", "gpu/global"),
        ("y", (100,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For(".blockIdx.x", 0, 3) as i:
            with ft.For(".threadIdx.x", 0, 34) as j:
                with ft.If(ft.any()):
                    ft.Any()
    assert ft.pop_ast().match(func.body)

    code = ft.codegen(func, target, verbose=True)
    x_np = np.array(range(0, 100), dtype="int32")
    y_np = np.zeros((100,), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array(range(1, 101), dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_not_remove_necessary_range_guard():

    @ft.transform
    def test(x, y):
        x: ft.Var[(5, 32), "int32", "input", "gpu/global"]
        y: ft.Var[(5, 32), "int32", "output", "gpu/global"]
        #! label: L1
        for i in range(0, 5):
            #! label: L2
            for j in range(0, 32):
                if i % 4 * 32 + j < 100:
                    y[i, j] = x[i, j] + 1

    s = ft.Schedule(test)
    s.parallelize('L1', "threadIdx.y")
    s.parallelize('L2', "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=2)

    with ft.VarDef([
        ("x", (5, 32), "int32", "input", "gpu/global"),
        ("y", (5, 32), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For(".blockIdx.x", 0, 5) as i:
            with ft.For(".threadIdx.x", 0, 32) as j:
                with ft.If(ft.any()):
                    ft.Any()
    assert ft.pop_ast().match(func.body)


def test_shmem():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4,), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        #! label: L1
        for i in range(0, 4):
            #! label: S1
            y[i] = x[i] + 1

    with ft.VarDef([
        ("x", (4,), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            ft.MarkLabel("S1")
            y[i] = x[i] + 1
    assert ft.pop_ast().match(test.body)

    with device:
        s = ft.Schedule(test)
        s.cache("S1", "x", "gpu/shared")
        s.parallelize("L1", "threadIdx.x")
        func = ft.lower(s.func(), verbose=1)
        code = ft.codegen(func, verbose=True)
        assert "__shared__" in code.code
        x_np = np.array([1, 2, 3, 4], dtype="int32")
        y_np = np.zeros((4,), dtype="int32")
        x_arr = ft.Array(x_np)
        y_arr = ft.Array(y_np)
        ft.build_binary(code)(x=x_arr, y=y_arr)
        y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_global_mem():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4,), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        t = ft.empty((4,), "int32", "gpu/global")
        #! label: L1
        for i in range(0, 4):
            t[i] = x[i] * 2
        #! label: L2
        for i in range(0, 4):
            y[i] = t[i] + 1

    with ft.VarDef([
        ("x", (4,), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.VarDef("t", (4,), "int32", "cache", "gpu/global") as t:
            with ft.For("i1", 0, 4, label="L1") as i:
                t[i] = x[i] * 2
            with ft.For("i2", 0, 4, label="L2") as i:
                y[i] = t[i] + 1
    assert ft.pop_ast().match(test.body)

    with device:
        s = ft.Schedule(test)
        s.parallelize("L1", "threadIdx.x")
        s.parallelize("L2", "threadIdx.x")
        func = ft.lower(s.func(), skip_passes=['prop_one_time_use'], verbose=1)
        code = ft.codegen(func, verbose=True)
        assert "__glmem +" in code.code  # offset from memory pool
        x_np = np.array([1, 2, 3, 4], dtype="int32")
        y_np = np.zeros((4,), dtype="int32")
        x_arr = ft.Array(x_np)
        y_arr = ft.Array(y_np)
        ft.build_binary(code)(x=x_arr, y=y_arr)
        y_np = y_arr.numpy()

    y_std = np.array([3, 5, 7, 9], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_global_mem_dynamic():

    @ft.transform
    def test(n, x, y):
        n: ft.Var[(), "int32", "input", "byvalue"]
        x: ft.Var[(n[...],), "int32", "input", "gpu/global"]
        y: ft.Var[(n[...],), "int32", "output", "gpu/global"]
        t = ft.empty((n[...],), "int32", "gpu/global")
        #! label: L1
        for i in range(0, n[...]):
            t[i] = x[i] * 2
        #! label: L2
        for i in range(0, n[...]):
            y[i] = t[i] + 1

    with device:
        s = ft.Schedule(test)
        s.parallelize("L1", "threadIdx.x")
        s.parallelize("L2", "threadIdx.x")
        func = ft.lower(s.func(), skip_passes=['prop_one_time_use'], verbose=1)
        code = ft.codegen(func, verbose=True)
        assert "cudaNewFromPool" in code.code
        n_np = np.array(4, dtype="int32")
        x_np = np.array([1, 2, 3, 4], dtype="int32")
        y_np = np.zeros((4,), dtype="int32")
        n_arr = ft.Array(n_np)
        x_arr = ft.Array(x_np)
        y_arr = ft.Array(y_np)
        ft.build_binary(code)(n=n_arr, x=x_arr, y=y_arr)
        y_np = y_arr.numpy()

    y_std = np.array([3, 5, 7, 9], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_global_mem_in_kernel():

    @ft.transform
    def test(x, y1, y2, z1, z2):
        x: ft.Var[(8,), "int32", "input", "gpu/global"]
        y1: ft.Var[(8,), "int32", "output", "gpu/global"]
        y2: ft.Var[(8,), "int32", "output", "gpu/global"]
        z1: ft.Var[(4,), "int32", "output", "gpu/global"]
        z2: ft.Var[(4,), "int32", "output", "gpu/global"]
        #! label: L1
        for i in range(0, 8):
            t = ft.empty((), "int32", "gpu/global")
            t[()] = x[i] * 2
            y1[i] = t[()] + 1
            y2[i] = t[()] + 2
        #! label: L2
        for i in range(0, 4):
            t = ft.empty((), "int32", "gpu/global")
            t[()] = x[i] * 3
            z1[i] = t[()] + 1
            z2[i] = t[()] + 2

    with device:
        s = ft.Schedule(test)
        s.parallelize("L1", "threadIdx.x")
        s.parallelize("L2", "threadIdx.x")
        func = ft.lower(s.func(), verbose=1)
        code = ft.codegen(func, verbose=True)
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype="int32")
        y1_np = np.zeros((8,), dtype="int32")
        y2_np = np.zeros((8,), dtype="int32")
        z1_np = np.zeros((4,), dtype="int32")
        z2_np = np.zeros((4,), dtype="int32")
        x_arr = ft.Array(x_np)
        y1_arr = ft.Array(y1_np)
        y2_arr = ft.Array(y2_np)
        z1_arr = ft.Array(z1_np)
        z2_arr = ft.Array(z2_np)
        ft.build_binary(code)(x_arr, y1_arr, y2_arr, z1_arr, z2_arr)
        y1_np = y1_arr.numpy()
        y2_np = y2_arr.numpy()
        z1_np = z1_arr.numpy()
        z2_np = z2_arr.numpy()

    y1_std = np.array([3, 5, 7, 9, 11, 13, 15, 17], dtype="int32")
    y2_std = np.array([4, 6, 8, 10, 12, 14, 16, 18], dtype="int32")
    z1_std = np.array([4, 7, 10, 13], dtype="int32")
    z2_std = np.array([5, 8, 11, 14], dtype="int32")
    assert np.array_equal(y1_np, y1_std)
    assert np.array_equal(y2_np, y2_std)
    assert np.array_equal(z1_np, z1_std)
    assert np.array_equal(z2_np, z2_std)


def test_pass_by_value_0d():

    @ft.transform
    def test(n, x, y):
        n: ft.Var[(), "int32", "input", "byvalue"]
        x: ft.Var[(n, 4), "int32", "input", "gpu/global"]
        y: ft.Var[(n, 4), "int32", "output", "gpu/global"]
        #! label: L1
        for i in range(0, 4):
            #! label: L2
            for j in range(0, n):
                y[j, i] = x[j, i] + 1

    with ft.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ft.VarDef([
            ("x", (n, 4), "int32", "input", "gpu/global"),
            ("y", (n, 4), "int32", "output", "gpu/global"),
        ]) as (x, y):
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", 0, n, label="L2") as j:
                    y[j, i] = x[j, i] + 1
    assert ft.pop_ast().match(test.body)

    with device:
        s = ft.Schedule(test)
        s.parallelize("L1", "threadIdx.x")
        func = ft.lower(s.func(), verbose=1)
        code = ft.codegen(func, verbose=True)
        n_np = np.array(5, dtype="int32")
        x_np = np.array([[1, 2, 3, 4]] * 5, dtype="int32")
        y_np = np.zeros((5, 4), dtype="int32")
        n_arr = ft.Array(n_np)
        x_arr = ft.Array(x_np)
        y_arr = ft.Array(y_np)
        ft.build_binary(code)(n=n_arr, x=x_arr, y=y_arr)
        y_np = y_arr.numpy()

    y_std = np.array([[2, 3, 4, 5]] * 5, dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_pass_by_value_1d():

    @ft.transform
    def test(n, x, y):
        n: ft.Var[(1,), "int32", "input", "byvalue"]
        x: ft.Var[(n[0], 4), "int32", "input", "gpu/global"]
        y: ft.Var[(n[0], 4), "int32", "output", "gpu/global"]
        #! label: L1
        for i in range(0, 4):
            #! label: L2
            for j in range(0, n[0]):
                y[j, i] = x[j, i] + 1

    with ft.VarDef("n", (1,), "int32", "input", "byvalue") as n:
        with ft.VarDef([
            ("x", (n[0], 4), "int32", "input", "gpu/global"),
            ("y", (n[0], 4), "int32", "output", "gpu/global"),
        ]) as (x, y):
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", 0, n[0], label="L2") as j:
                    y[j, i] = x[j, i] + 1
    assert ft.pop_ast().match(test.body)

    with device:
        s = ft.Schedule(test)
        s.parallelize("L1", "threadIdx.x")
        func = ft.lower(s.func(), verbose=1)
        code = ft.codegen(func, verbose=True)
        n_np = np.array([5], dtype="int32")
        x_np = np.array([[1, 2, 3, 4]] * 5, dtype="int32")
        y_np = np.zeros((5, 4), dtype="int32")
        n_arr = ft.Array(n_np)
        x_arr = ft.Array(x_np)
        y_arr = ft.Array(y_np)
        ft.build_binary(code)(n=n_arr, x=x_arr, y=y_arr)
        y_np = y_arr.numpy()

    y_std = np.array([[2, 3, 4, 5]] * 5, dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_dynamic_2d_array():
    with ft.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ft.VarDef([
            ("x", (n, n), "int32", "input", "gpu/global"),
            ("y", (n, n), "int32", "output", "gpu/global"),
        ]) as (x, y):
            with ft.For("i", 0, n, label="L1") as i:
                with ft.For("j", 0, n, label="L2") as j:
                    y[i, j] = x[i, j] + 1

    with device:
        s = ft.Schedule(ft.Func("main", ["n", "x", "y"], [], ft.pop_ast()))
        outer, inner = s.split("L1", 4)
        s.reorder([inner, outer])
        s.parallelize(inner, "threadIdx.x")
        func = ft.lower(s.func(), verbose=1)
        code = ft.codegen(func, verbose=True)
        n_np = np.array(5, dtype="int32")
        x_np = np.random.randint(0, 100, (5, 5)).astype("int32")
        y_np = np.zeros((5, 5), dtype="int32")
        n_arr = ft.Array(n_np)
        x_arr = ft.Array(x_np)
        y_arr = ft.Array(y_np)
        ft.build_binary(code)(n=n_arr, x=x_arr, y=y_arr)
        y_np = y_arr.numpy()

    y_std = x_np + 1
    assert np.array_equal(y_np, y_std)


def test_dynamic_thread_dim_1():
    with ft.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ft.VarDef([
            ("x", (n, n), "int32", "input", "gpu/global"),
            ("y", (n, n), "int32", "output", "gpu/global"),
        ]) as (x, y):
            with ft.For("i", 0, n, label="L1") as i:
                with ft.For("j", 0, n, label="L2") as j:
                    y[i, j] = x[i, j] + 1

    with device:
        s = ft.Schedule(ft.Func("main", ["n", "x", "y"], [], ft.pop_ast()))
        s.parallelize("L1", "threadIdx.x")
        func = ft.lower(s.func(), verbose=1)
        code = ft.codegen(func, verbose=True)
        n_np = np.array(5, dtype="int32")
        x_np = np.random.randint(0, 100, (5, 5)).astype("int32")
        y_np = np.zeros((5, 5), dtype="int32")
        n_arr = ft.Array(n_np)
        x_arr = ft.Array(x_np)
        y_arr = ft.Array(y_np)
        ft.build_binary(code)(n=n_arr, x=x_arr, y=y_arr)
        y_np = y_arr.numpy()

    y_std = x_np + 1
    assert np.array_equal(y_np, y_std)


def test_dynamic_thread_dim_2():

    @ft.transform()
    def func(x, offset, y):
        x: ft.Var[(27, 256), "float32", "input", "gpu/global"]
        offset: ft.Var[(28,), "int32", "input", "byvalue"]
        y: ft.Var[(256,), "float32", "output", "gpu/global"]

        #! label: Li
        for i in range(27):
            cur_start = offset[i]
            cur_end = offset[i + 1]
            #! label: La
            for a in range(cur_start, cur_end):
                #! label: Loc
                for oc in range(256):
                    y[oc % 2] += x[i, oc]

    s = ft.Schedule(func)
    by, la = s.split("La", factor=4)
    bx, loc = s.split("Loc", factor=4)
    s.reorder([by, bx, la, loc])
    s.parallelize(by, "blockIdx.y")
    s.parallelize(bx, "blockIdx.x")
    s.parallelize(la, "threadIdx.y")
    s.parallelize(loc, "threadIdx.x")
    func = s.func()
    func = ft.lower(func, target, verbose=1)

    with ft.VarDef([("x", (27, 256), "float32", "input", "gpu/global"),
                    ("offset", (28,), "int32", "input", "byvalue"),
                    ("y", (256,), "float32", "output", "gpu/global")
                   ]) as (x, offset, y):
        with ft.For("i", 0, 27) as i:
            with ft.For("blockIdx.y", 0, ft.any()) as by:
                with ft.For("blockIdx.x", 0, 64) as bx:
                    with ft.For("threadIdx.y", 0,
                                ft.min(offset[i + 1] + -1 * offset[i],
                                       4)) as ty:
                        with ft.If(ft.unbound(ty < ft.any())):
                            ft.Any()
    assert ft.pop_ast().match(func.body)


def test_use_cpu_iters():
    with ft.VarDef("y", (4, 1000), "int32", "output", "gpu/global") as y:
        with ft.For("i", 0, 4, label="Li") as i:
            with ft.For("j", 0, 1000, label="Lj") as j:
                y[i, j] = i + j

    with device:
        s = ft.Schedule(ft.Func("main", ["y"], [], ft.pop_ast()))
        s.parallelize('Lj', "threadIdx.x")
        func = ft.lower(s.func(), verbose=1)
        code = ft.codegen(func, verbose=True)
        y_np = np.zeros((4, 1000), dtype="int32")
        y_arr = ft.Array(y_np)
        ft.build_binary(code)(y_arr)
        y_np = y_arr.numpy()

    y_std = np.array([[i + j for j in range(1000)] for i in range(4)],
                     dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_dynamic_thread_dim_by_cpu_iters():
    with ft.VarDef([("n", (4,), "int32", "input", "byvalue"),
                    ("y", (4, 20), "int32", "inout", "gpu/global")]) as (n, y):
        with ft.For("i", 0, 4, label="Li") as i:
            with ft.For("j", 0, n[i], label="Lj") as j:
                y[i, j] = i * j

    with device:
        s = ft.Schedule(ft.Func("main", ["n", "y"], [], ft.pop_ast()))
        s.parallelize('Lj', "threadIdx.x")
        func = ft.lower(s.func(), verbose=1)
        code = ft.codegen(func, verbose=True)
        n_np = np.array([5, 10, 15, 20], dtype="int32")
        y_np = np.zeros((4, 20), dtype="int32")
        n_arr = ft.Array(n_np)
        y_arr = ft.Array(y_np)
        ft.build_binary(code)(n_arr, y_arr)
        y_np = y_arr.numpy()

    y_std = np.array([[i * j if j < n else 0
                       for j in range(20)]
                      for i, n in enumerate([5, 10, 15, 20])],
                     dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_intrinsic():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4,), "float32", "input", "gpu/global"]
        y: ft.Var[(4,), "float32", "output", "gpu/global"]
        #! label: L1
        for i in range(0, 4):
            y[i] = ft.intrinsic("sinf(%)", x[i], ret_type="float32")

    with ft.VarDef([
        ("x", (4,), "float32", "input", "gpu/global"),
        ("y", (4,), "float32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            y[i] = ft.intrinsic("sinf(%)", x[i], ret_type="float32")
    assert ft.pop_ast().match(test.body)

    with device:
        s = ft.Schedule(test)
        s.parallelize("L1", "threadIdx.x")
        func = ft.lower(s.func(), verbose=1)
        code = ft.codegen(func, verbose=True)
        x_np = np.array([1, 2, 3, 4], dtype="float32")
        y_np = np.zeros((4,), dtype="float32")
        x_arr = ft.Array(x_np)
        y_arr = ft.Array(y_np)
        ft.build_binary(code)(x=x_arr, y=y_arr)
        y_np = y_arr.numpy()

    y_std = np.array(np.sin(x_np), dtype="float32")
    assert np.all(np.isclose(y_np, y_std))


def test_dynamic_shared_memory_size():
    with ft.VarDef("n", (), "int32", "input", "gpu/global") as n:
        with ft.VarDef([
            ("x", (4, 256), "int32", "input", "gpu/global"),
            ("y", (4, 256), "int32", "output", "gpu/global"),
        ]) as (x, y):
            with ft.For("i", 0, 4, label="L0") as i:
                with ft.Assert(n <= 256):
                    with ft.VarDef("t", (n,), "int32", "cache",
                                   "gpu/shared") as t:
                        with ft.For("j", 0, n, label="L1") as j:
                            t[j] = x[i, j] * 2
                        with ft.For("j", 0, n, label="L2") as j:
                            y[i, j] = t[j] + 1
                        with ft.For("j", n, 256, label="L3") as j:
                            y[i, j] = 0

    s = ft.Schedule(ft.Func("main", ["n", "x", "y"], [], ft.pop_ast()))
    s.parallelize("L0", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    skip_passes=['prop_one_time_use'],
                    verbose=1)

    with ft.VarDef("n", (), "int32", "input", "gpu/global") as n:
        with ft.VarDef([
            ("x", (4, 256), "int32", "input", "gpu/global"),
            ("y", (4, 256), "int32", "output", "gpu/global"),
        ]) as (x, y):
            with ft.For(".threadIdx.y", 0, 4) as i:
                with ft.Assert(n <= 256):
                    with ft.VarDef("t", (4, ft.max(n - 1, 0) + 1), "int32",
                                   "cache", "gpu/shared") as t:
                        with ft.For("j", 0, n) as j:
                            t[i, j] = x[i, j] * 2
                        with ft.For("j", 0, n) as j:
                            y[i, j] = t[i, j] + 1
                    with ft.For("j", n, 256) as j:
                        y[i, j] = 0
    assert ft.pop_ast().match(func.body)

    code = ft.codegen(func, target, verbose=True)
    n_np = np.array(200, dtype="int32")
    x_np = np.array([range(256)] * 4, dtype="int32")
    y_np = np.zeros((4, 256), dtype="int32")
    n_arr = ft.Array(n_np)
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(n=n_arr, x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([list(range(1, 401, 2)) + [0] * 56] * 4, dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_parallel_different_length():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(4, 4), "int32", "input", "gpu/global"]
        b: ft.Var[(4, 8), "int32", "input", "gpu/global"]
        c: ft.Var[(4, 8), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((4,), "int32", "gpu/shared")
            #! label: L1
            for j in range(0, 4):
                t[j] = a[i, j]
            #! label: L2
            for j in range(0, 4):
                #! label: L3
                for k in range(0, 8):
                    c[i, k] = c[i, k] + t[j] * b[j, k]

    with ft.VarDef([
        ("a", (4, 4), "int32", "input", "gpu/global"),
        ("b", (4, 8), "int32", "input", "gpu/global"),
        ("c", (4, 8), "int32", "output", "gpu/global"),
    ]) as (a, b, c):
        with ft.For("i", 0, 4, label="L0") as i:
            with ft.VarDef("t", (4,), "int32", "cache", "gpu/shared") as t:
                with ft.For("j1", 0, 4, label="L1") as j:
                    t[j] = a[i, j]
                with ft.For("j2", 0, 4, label="L2") as j:
                    with ft.For("k", 0, 8, label="L3") as k:
                        c[i, k] = c[i, k] + t[j] * b[j, k]
    assert ft.pop_ast().match(test.body)

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L3", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    skip_passes=['prop_one_time_use'],
                    verbose=1)

    with ft.VarDef([
        ("a", (4, 4), "int32", "input", "gpu/global"),
        ("b", (4, 8), "int32", "input", "gpu/global"),
        ("c", (4, 8), "int32", "output", "gpu/global"),
    ]) as (a, b, c):
        with ft.For(".blockIdx.x", 0, 4) as blk:
            with ft.For(".threadIdx.x", 0, 8) as th:
                with ft.VarDef("t", (4,), "int32", "cache", "gpu/shared") as t:
                    with ft.If(th < 4):
                        t[th] = a[blk, th]
                    ft.Eval(
                        ft.intrinsic("__syncwarp(__activemask())",
                                     has_side_effect=True))
                    with ft.For("j", 0, 4) as j:
                        ft.Any()
    assert ft.pop_ast().match(func.body)

    code = ft.codegen(func, target, verbose=True)
    a_np = np.random.randint(0, 100, (4, 4)).astype("int32")
    b_np = np.random.randint(0, 100, (4, 8)).astype("int32")
    c_np = np.zeros((4, 8), dtype="int32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np)
    ft.build_binary(code, device)(a=a_arr, b=b_arr, c=c_arr)
    c_np = c_arr.numpy()

    c_std = a_np @ b_np
    assert np.array_equal(c_np, c_std)


def test_bounded_length():

    @ft.transform
    def test(a, b):
        a: ft.Var[(100, 100), "int32", "input", "gpu/global"]
        b: ft.Var[(100, 100), "int32", "output", "gpu/global"]
        #! label: Li
        for i in range(100):
            #! label: Lj
            for j in range(i):
                b[i, j] = a[i, j] + 1

    s = ft.Schedule(test)
    s.parallelize("Lj", "threadIdx.y")
    s.parallelize("Li", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    with ft.VarDef([
        ("a", (100, 100), "int32", "input", "gpu/global"),
        ("b", (100, 100), "int32", "output", "gpu/global"),
    ]) as (a, b):
        with ft.For(".threadIdx.y", 0, 99) as thy:
            with ft.For(".threadIdx.x", 0, 99) as thx:  # i = thx + 1
                with ft.If(ft.unbound(thy < thx + 1)):
                    b[thx + 1, thy] = a[thx + 1, thy] + 1
    assert ft.pop_ast().match(func.body)


def test_parallel_broadcast():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(4, 1), "int32", "input", "gpu/global"]
        b: ft.Var[(1, 8), "int32", "input", "gpu/global"]
        c: ft.Var[(4, 8), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((1,), "int32", "gpu/shared")
            t[0] = a[i, 0]
            #! label: L1
            for k in range(0, 8):
                c[i, k] = c[i, k] + t[0] * b[0, k]

    with ft.VarDef([
        ("a", (4, 1), "int32", "input", "gpu/global"),
        ("b", (1, 8), "int32", "input", "gpu/global"),
        ("c", (4, 8), "int32", "output", "gpu/global"),
    ]) as (a, b, c):
        with ft.For("i", 0, 4, label="L0") as i:
            with ft.VarDef("t", (1,), "int32", "cache", "gpu/shared") as t:
                t[0] = a[i, 0]
                with ft.For("k", 0, 8, label="L1") as k:
                    c[i, k] = c[i, k] + t[0] * b[0, k]
    assert ft.pop_ast().match(test.body)

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    skip_passes=['prop_one_time_use'],
                    verbose=1)

    with ft.VarDef([
        ("a", (4, 1), "int32", "input", "gpu/global"),
        ("b", (1, 8), "int32", "input", "gpu/global"),
        ("c", (4, 8), "int32", "output", "gpu/global"),
    ]) as (a, b, c):
        with ft.For(".blockIdx.x", 0, 4) as blk:
            with ft.For(".threadIdx.x", 0, 8) as th:
                with ft.VarDef("t", (1,), "int32", "cache", "gpu/shared") as t:
                    with ft.If(th == 0):
                        t[0] = a[blk, 0]
                    ft.Eval(
                        ft.intrinsic("__syncwarp(__activemask())",
                                     has_side_effect=True))
                    ft.Any()
    assert ft.pop_ast().match(func.body)

    code = ft.codegen(func, target, verbose=True)
    a_np = np.random.randint(0, 100, (4, 1)).astype("int32")
    b_np = np.random.randint(0, 100, (1, 8)).astype("int32")
    c_np = np.zeros((4, 8), dtype="int32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np)
    ft.build_binary(code, device)(a=a_arr, b=b_arr, c=c_arr)
    c_np = c_arr.numpy()

    c_std = a_np @ b_np
    assert np.array_equal(c_np, c_std)


def test_unbounded_length():

    @ft.transform
    def test(n, x, y):
        n: ft.Var[(), "int32", "input", "gpu/global"]
        x: ft.Var[(n,), "int32", "input", "gpu/global"]
        y: ft.Var[(n,), "int32", "output", "gpu/global"]
        #! label: L1
        for i in range(0, n):
            y[i] = x[i] + 1

    with ft.VarDef("n", (), "int32", "input", "gpu/global") as n:
        with ft.VarDef([
            ("x", (n,), "int32", "input", "gpu/global"),
            ("y", (n,), "int32", "output", "gpu/global"),
        ]) as (x, y):
            with ft.For("i", 0, n, label="L1") as i:
                y[i] = x[i] + 1
    assert ft.pop_ast().match(test.body)

    s = ft.Schedule(test)
    s.parallelize("L1", "threadIdx.x")
    with pytest.raises(ft.InvalidProgram):
        ft.lower(s.func(), target, verbose=1)


def test_unroll_for():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 64), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        #! label: L1
        for i in range(0, 4):
            #! label: L2
            for j in range(0, 64):
                y[i] = y[i] + x[i, j]

    with ft.VarDef([
        ("x", (4, 64), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 64, label="L2") as j:
                y[i] = y[i] + x[i, j]
    assert ft.pop_ast().match(test.body)

    s = ft.Schedule(test)
    s.parallelize("L1", "blockIdx.x")
    s.unroll("L2")
    func = ft.lower(s.func(), target, verbose=1)

    code = ft.codegen(func, target, verbose=True)
    assert "atomicAdd" not in code.code
    assert "+=" in code.code
    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.sum(x_np, axis=1)
    assert np.array_equal(y_np, y_std)


def test_streams():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 256), "int32", "output", "gpu/global"]
        #! label: L1
        for i in range(0, 4):
            #! label: L2
            for j in range(0, 256):
                y[i, j] = x[i, j] + 1
            #! label: L3
            for j in range(0, 128):
                y[i, j * 2] *= 2

    s = ft.Schedule(test)
    s.parallelize("L1", "cudaStream")
    s.parallelize("L2", "threadIdx.x")
    s.parallelize("L3", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cudaStreamCreate" in code.code
    x_np = np.random.randint(0, 100, (4, 256)).astype("int32")
    y_np = np.zeros((4, 256), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = ((x_np + 1).reshape(4, 128, 2) * [2, 1]).reshape(4, 256)
    assert np.array_equal(y_np, y_std)


def test_merge_no_deps_1():

    @ft.transform(verbose=1)
    def test(ptr, edge1, edge2):
        ptr: ft.Var[(4, 11), "int32", "input", "cpu"]
        edge1: ft.Var[(4, 50), "int32", "input", "cpu"]
        edge2: ft.Var[(4, 50), "int32", "output", "cpu"]
        #! label: Lb
        for b in range(4):
            #! label: Li1
            #! no_deps: edge2
            for i in range(10):
                for j in range(ptr[b, i], ptr[b, i + 1]):
                    edge2[b, j] = edge1[b, j] + i
            #! label: Li2
            #! no_deps: edge2
            for i in range(10):
                for j in range(ptr[b, i], ptr[b, i + 1]):
                    edge2[b, j] += j

    s = ft.Schedule(test)
    s.parallelize("Lb", "blockIdx.x")
    s.parallelize("Li1", "threadIdx.x")
    s.parallelize("Li2", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    def matcher(x):
        if x.type() == ft.ASTNodeType.For:
            node = x
            if str(node.property.parallel) == "threadIdx.x":
                return True
        return False

    checker = ft.Schedule(func)
    assert checker.find(matcher).property.no_deps == ['edge2']


def test_merge_no_deps_2():

    @ft.transform(verbose=1)
    def test(ptr, edge1, edge2):
        ptr: ft.Var[(4, 11), "int32", "input", "cpu"]
        edge1: ft.Var[(4, 50), "int32", "input", "cpu"]
        edge2: ft.Var[(4, 50), "int32", "output", "cpu"]
        foobar: ft.Var((
            4,
            10,
        ), "int32", "output", "cpu")
        #! label: Lb
        for b in range(4):
            #! label: Li1
            #! no_deps: edge2
            for i in range(10):
                for j in range(ptr[b, i], ptr[b, i + 1]):
                    edge2[b, j] = edge1[b, j] + i
            #! label: Li2
            for i in range(10):
                # Nothing to do with edge2 here
                foobar[b, i] = i

    s = ft.Schedule(test)
    s.parallelize("Lb", "blockIdx.x")
    s.parallelize("Li1", "threadIdx.x")
    s.parallelize("Li2", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    def matcher(x):
        if x.type() == ft.ASTNodeType.For:
            node = x
            if str(node.property.parallel) == "threadIdx.x":
                return True
        return False

    checker = ft.Schedule(func)
    assert checker.find(matcher).property.no_deps == ['edge2']


def test_merge_no_deps_3():

    @ft.transform(verbose=1)
    def test(ptr, edge1, edge2):
        ptr: ft.Var[(4, 11), "int32", "input", "cpu"]
        edge1: ft.Var[(4, 50), "int32", "input", "cpu"]
        edge2: ft.Var[(4, 50), "int32", "output", "cpu"]
        #! label: Lb
        for b in range(4):
            #! label: Li1
            #! no_deps: edge2
            for i in range(10):
                for j in range(ptr[b, i], ptr[b, i + 1]):
                    edge2[b, j] = edge1[b, j] + i
            #! label: Li2  # If we don't mark edge2 here
            for i in range(10):
                for j in range(ptr[b, i], ptr[b, i + 1] + 1):
                    edge2[b, j] = edge2[b, j] * 2 + j

    s = ft.Schedule(test)
    s.parallelize("Lb", "blockIdx.x")
    s.parallelize("Li1", "threadIdx.x")
    with pytest.raises(ft.InvalidSchedule):
        s.parallelize("Li2", "threadIdx.x")


def test_access_gpu_from_cpu_for_debugging():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4,), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        #! label: L1
        for i in range(0, 4):
            y[i] = x[i] + 1

    with ft.VarDef([
        ("x", (4,), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            y[i] = x[i] + 1
    assert ft.pop_ast().match(test.body)

    s = ft.Schedule(test)
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_error_invalid_mtype_out_of_kernel():

    with device:

        @ft.lower
        @ft.transform
        def test(x, y):
            x: ft.Var[(4,), "int32", "input", "gpu/local"]
            y: ft.Var[(4,), "int32", "output", "gpu/local"]
            #! label: L1
            for i in range(0, 4):
                y[i] = x[i] + 1

        with pytest.raises(ft.InvalidProgram):
            ft.codegen(test)


def test_error_invalid_mtype_inside_kernel():

    with device:

        @ft.lower
        @ft.schedule(callback=lambda s: s.parallelize("L1", "threadIdx.x"))
        @ft.transform
        def test(x, y):
            x: ft.Var[(4,), "int32", "input", "cpu"]
            y: ft.Var[(4,), "int32", "output", "cpu"]
            #! label: L1
            for i in range(0, 4):
                y[i] = x[i] + 1

        with pytest.raises(ft.InvalidProgram):
            ft.codegen(test)
