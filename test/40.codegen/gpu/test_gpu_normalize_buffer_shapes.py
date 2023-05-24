'''
This file tests GPU-specific passes that adjust the shapes of `VarDef`s, especially
`pass/gpu/multiplex_buffers` and `pass/gpu/simplex_buffers`. These `VarDef`s are
utilized by CodeGenCUDA to produce CUDA arrays or buffers, which should accurately
denote whether they are shared among or local to threads or blocks.
'''

import freetensor as ft
import numpy as np
import pytest

if not ft.with_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

device = ft.GPU()
target = device.target()


def test_multiplex_shared_basic():
    '''
    `t` should contain different data for different `i`
    '''

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 256), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((256,), "int32", "gpu/shared")
            #! label: L1
            for j in range(0, 256):
                t[j] = x[i, j] * 2
            #! label: L2
            for j in range(0, 256):
                y[i, j] = t[j] + 1

    with ft.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, label="L0") as i:
            with ft.VarDef("t", (256,), "int32", "cache", "gpu/shared") as t:
                with ft.For("j1", 0, 256, label="L1") as j:
                    t[j] = x[i, j] * 2
                with ft.For("j2", 0, 256, label="L2") as j:
                    y[i, j] = t[j] + 1
    assert ft.pop_ast().match(test.body)

    s = ft.Schedule(test)
    s.parallelize("L0", "threadIdx.y")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    skip_passes=['prop_one_time_use'],
                    verbose=1)

    with ft.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For(".threadIdx.y", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 256) as j:
                with ft.VarDef("t", (4, 256), "int32", "cache",
                               "gpu/shared") as t:
                    t[i, j] = x[i, j] * 2
                    y[i, j] = t[i, j] + 1
    assert ft.pop_ast().match(func.body)

    code = ft.codegen(func, target, verbose=True)
    x_np = np.array([range(256)] * 4, dtype="int32")
    y_np = np.zeros((4, 256), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([range(1, 513, 2)] * 4, dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_multiplex_no_dependence():
    '''
    `t` is already non-overlapping for different `i`, so no need to touch it
    '''

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 256), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((256,), "int32", "gpu/shared")
            #! label: L1
            for j in range(i * 64, (i + 1) * 64):
                t[j] = x[i, j] * 2
                # No need to hoist over i, although i is not present here
            #! label: L2
            for j in range(i * 64, (i + 1) * 64):
                y[i, j] = t[j] + 1

    s = ft.Schedule(test)
    s.parallelize("L0", "threadIdx.y")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    skip_passes=['prop_one_time_use'],
                    verbose=1)

    with ft.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For(".threadIdx.y", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 64) as j:
                with ft.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    t[j + i * 64] = x[i, j + i * 64] * 2
                    y[i, j + i * 64] = t[j + i * 64] + 1
    assert ft.pop_ast().match(func.body)


def test_multiplex_shared_loop_invariant():
    '''
    `t` does not have to contain different data for different `i`, because
    data for all `i` is the same
    '''

    @ft.transform
    def test(x, y):
        x: ft.Var[(256,), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 256), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((256,), "int32", "gpu/shared")
            #! label: L1
            for j in range(0, 256):
                t[j] = x[j] * 2
            #! label: L2
            for j in range(0, 256):
                y[i, j] = t[j] * i

    s = ft.Schedule(test)
    s.parallelize("L0", "threadIdx.y")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    skip_passes=['prop_one_time_use'],
                    verbose=1)

    with ft.VarDef([
        ("x", (256,), "int32", "input", "gpu/global"),
        ("y", (4, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For(".threadIdx.y", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 256) as j:
                with ft.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    t[j] = x[j] * 2
                    ft.Eval(
                        ft.intrinsic("__syncthreads()", has_side_effect=True))
                    y[i, j] = t[j] * i
    assert ft.pop_ast().match(func.body)

    code = ft.codegen(func, target, verbose=True)
    x_np = np.array(range(256), dtype="int32")
    y_np = np.zeros((4, 256), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([
        np.array(range(256)) * 0,
        np.array(range(256)) * 2,
        np.array(range(256)) * 4,
        np.array(range(256)) * 6
    ],
                     dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_multiplex_global_loop_invariant():
    '''
    `t` does not have to contain different data for different `i`, because
    data for all `i` is the same
    '''

    @ft.transform
    def test(x, y):
        x: ft.Var[(256,), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 256), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((256,), "int32", "gpu/global")
            #! label: L1
            for j in range(0, 256):
                t[j] = x[j] * 2
            #! label: L2
            for j in range(0, 256):
                y[i, j] = t[j] * i

    s = ft.Schedule(test)
    s.parallelize("L0", "threadIdx.y")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    skip_passes=['prop_one_time_use'],
                    verbose=1)

    with ft.VarDef([
        ("x", (256,), "int32", "input", "gpu/global"),
        ("y", (4, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.VarDef("t", (256,), "int32", "cache", "gpu/global") as t:
            with ft.For(".threadIdx.y", 0, 4) as i:
                with ft.For(".threadIdx.x", 0, 256) as j:
                    t[j] = x[j] * 2
                    ft.Eval(
                        ft.intrinsic("__syncthreads()", has_side_effect=True))
                    y[i, j] = t[j] * i
    assert ft.pop_ast().match(func.body)

    code = ft.codegen(func, target, verbose=True)
    x_np = np.array(range(256), dtype="int32")
    y_np = np.zeros((4, 256), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([
        np.array(range(256)) * 0,
        np.array(range(256)) * 2,
        np.array(range(256)) * 4,
        np.array(range(256)) * 6
    ],
                     dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_multiplex_global_loop_invariant_but_still_multiplex_across_blocks():
    '''
    `t` does not have to contain different data for different `i`, but we still
    need to allocate different parts of `t` for different `i`, because we cannot
    communiate across blocks.
    '''

    @ft.transform
    def test(x, y):
        x: ft.Var[(256,), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 256), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((256,), "int32", "gpu/global")
            #! label: L1
            for j in range(0, 256):
                t[j] = x[j] * 2
            #! label: L2
            for j in range(0, 256):
                y[i, j] = t[j] * i

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    skip_passes=['prop_one_time_use'],
                    verbose=1)

    with ft.VarDef([
        ("x", (256,), "int32", "input", "gpu/global"),
        ("y", (4, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.VarDef("t", (4, 256), "int32", "cache", "gpu/global") as t:
            with ft.For(".threadIdx.y", 0, 4) as i:
                with ft.For(".threadIdx.x", 0, 256) as j:
                    t[i, j] = x[j] * 2
                    # No sync here because we can't
                    y[i, j] = t[i, j] * i
    assert ft.pop_ast().match(func.body)

    code = ft.codegen(func, target, verbose=True)
    x_np = np.array(range(256), dtype="int32")
    y_np = np.zeros((4, 256), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([
        np.array(range(256)) * 0,
        np.array(range(256)) * 2,
        np.array(range(256)) * 4,
        np.array(range(256)) * 6
    ],
                     dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_multiplex_shared_not_loop_invariant():
    '''
    `t` should contain different data for different `i`. Although the first
    assignment to `t` assigns the same data, but the data assigned later is
    different across `i`.
    '''

    @ft.transform
    def test(x, y):
        x: ft.Var[(256,), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 256), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((256,), "int32", "gpu/shared")
            #! label: L1
            for j in range(0, 256):
                t[j] = 0
                if i < 2:
                    t[j] = x[j] * 2
            #! label: L2
            for j in range(0, 256):
                y[i, j] = t[j] + 1

    s = ft.Schedule(test)
    s.parallelize("L0", "threadIdx.y")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    skip_passes=['prop_one_time_use'],
                    verbose=1)

    @ft.transform
    def expected(x, y):
        x: ft.Var[(256,), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 256), "int32", "output", "gpu/global"]
        for i in range(0, 4):
            for j in range(0, 256):
                t = ft.empty((4, 256), "int32", "gpu/shared")
                t[i, j] = 0
                if i < 2:
                    t[i, j] = x[j] * 2
                y[i, j] = t[i, j] + 1

    assert expected.body.match(func.body)


def test_simplex_local_1():

    @ft.transform
    def test(x, y, z):
        x: ft.Var[(10, 10, 10), "int32", "input", "gpu/global"]
        y: ft.Var[(10, 10, 10), "int32", "output", "gpu/global"]
        z: ft.Var[(10, 10, 10), "int32", "output", "gpu/global"]
        #! label: Lb
        for b in range(10):
            #! label: t
            t = ft.empty((10, 10), "int32", "gpu/global")
            #! label: L0
            for i in range(10):
                for j in range(10):
                    t[i, j] = x[b, i, j] * 2
            #! label: L1
            for i in range(10):
                for j in range(10):
                    y[b, i, j] = t[i, j] + 1
            #! label: L2
            for i in range(10):
                for j in range(10):
                    z[b, i, j] = t[i, j] + 2

    s = ft.Schedule(test)
    s.parallelize("Lb", "blockIdx.x")
    s.parallelize("L0", "threadIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    s.set_mem_type("t", "gpu/local")
    func = ft.lower(s.func(), target, verbose=1)

    with ft.VarDef([("x", (10, 10, 10), "int32", "input", "gpu/global"),
                    ("y", (10, 10, 10), "int32", "output", "gpu/global"),
                    ("z", (10, 10, 10), "int32", "output", "gpu/global")
                   ]) as (x, y, z):
        with ft.For(".blockIdx", 0, 10) as b:
            with ft.For(".threadIdx.x", 0, 10) as i:
                with ft.VarDef("t", (1, 10), "int32", "cache",
                               "gpu/local") as t:
                    with ft.For("j", 0, 10) as j:
                        t[0, j] = x[b, i, j] * 2
                    with ft.For("j$1", 0, 10) as j:
                        y[b, i, j] = t[0, j] + 1
                    with ft.For("j$2", 0, 10) as j:
                        z[b, i, j] = t[0, j] + 2
    assert ft.pop_ast().match(func.body)

    code = ft.codegen(func, target, verbose=True)
    x_np = np.random.randint(0, 100, (10, 10, 10)).astype("int32")
    y_np = np.zeros((10, 10, 10), dtype="int32")
    z_np = np.zeros((10, 10, 10), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    z_arr = ft.Array(z_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr, z=z_arr)
    y_np = y_arr.numpy()
    z_np = z_arr.numpy()

    assert np.array_equal(y_np, x_np * 2 + 1)
    assert np.array_equal(z_np, x_np * 2 + 2)


def test_simplex_local_2():

    @ft.transform
    def test(x, y, z):
        x: ft.Var[(10, 10, 10), "int32", "input", "gpu/global"]
        y: ft.Var[(10, 10, 10), "int32", "output", "gpu/global"]
        #! label: Lb
        for b in range(10):
            #! label: t
            t = ft.empty((10, 10), "int32", "gpu/global")
            #! label: L0
            for i in range(10):
                for j in range(10):
                    t[i, j] = x[b, i, j] * 2
                for j in range(10):
                    t[i, j] += t[i, i]
                    # The last dimension can be removed although accessed with i
            #! label: L1
            for i in range(10):
                for j in range(10):
                    y[b, i, j] = t[i, j] + 1

    s = ft.Schedule(test)
    s.parallelize("Lb", "blockIdx.x")
    s.parallelize("L0", "threadIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.set_mem_type("t", "gpu/local")
    func = ft.lower(s.func(), target, verbose=1)

    with ft.VarDef([
        ("x", (10, 10, 10), "int32", "input", "gpu/global"),
        ("y", (10, 10, 10), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For(".blockIdx", 0, 10) as b:
            with ft.For(".threadIdx.x", 0, 10) as i:
                with ft.VarDef("t", (1, 10), "int32", "cache",
                               "gpu/local") as t:
                    with ft.For("j", 0, 10) as j:
                        t[0, j] = x[b, i, j] * 2
                    with ft.For("j$1", 0, 10) as j:
                        t[0, j] += t[0, i]
                    with ft.For("j$2", 0, 10) as j:
                        y[b, i, j] = t[0, j] + 1
    assert ft.pop_ast().match(func.body)
