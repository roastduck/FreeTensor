import freetensor as ft
from freetensor import debug
import pytest
import numpy as np

if not ft.with_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

device = ft.GPU()
target = device.target()


def test_syncthreads():

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
                y[i, j] = t[255 - j] + 1

    with ft.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, label="L0") as i:
            with ft.VarDef("t", (256,), "int32", "cache", "gpu/shared") as t:
                with ft.For("j1", 0, 256, label="L1") as j:
                    t[j] = x[i, j] * 2
                with ft.For("j2", 0, 256, label="L2") as j:
                    y[i, j] = t[255 - j] + 1
    assert ft.pop_ast().match(test.body)

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
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
        with ft.For(".blockIdx.x", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 256) as j:
                with ft.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    ft.Any()
                    ft.Eval(
                        ft.intrinsic("__syncthreads()", has_side_effect=True))
                    ft.Any()
    assert ft.pop_ast().match(func.body)

    code = ft.codegen(func, target, verbose=True)
    x_np = np.array([range(256)] * 4, dtype="int32")
    y_np = np.zeros((4, 256), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([range(511, -1, -2)] * 4, dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_syncthreads_in_loop():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 5, 256), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            for p in range(0, 5):
                t = ft.empty((256,), "int32", "gpu/shared")
                #! label: L1
                for j in range(0, 256):
                    t[j] = x[i, j] * p
                #! label: L2
                for j in range(0, 256):
                    y[i, p, j] = t[255 - j] + 1

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    skip_passes=['prop_one_time_use'],
                    verbose=1)

    with ft.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4, 5, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For(".blockIdx.x", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 256) as j:
                with ft.For("p", 0, 5) as p:
                    with ft.VarDef("t", (256,), "int32", "cache",
                                   "gpu/shared") as t:
                        ft.Any()
                        ft.Eval(
                            ft.intrinsic("__syncthreads()",
                                         has_side_effect=True))
                        ft.Any()
                    ft.Eval(
                        ft.intrinsic("__syncthreads()", has_side_effect=True))
    assert ft.pop_ast().match(func.body)


def test_syncthreads_at_outer_loop():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 5, 256), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((256,), "int32", "gpu/shared")
            #! label: L1
            for j in range(0, 256):
                t[j] = x[i, j]
            for p in range(0, 5):
                #! label: L2
                for j in range(0, 256):
                    y[i, p, j] = t[255 - j] + 1

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    skip_passes=['prop_one_time_use'],
                    verbose=1)

    with ft.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4, 5, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For(".blockIdx.x", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 256) as j:
                with ft.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    ft.Any()
                    ft.Eval(
                        ft.intrinsic("__syncthreads()",
                                     has_side_effect=True))  # Here outside p
                    with ft.For("p", 0, 5) as p:
                        ft.Any()
    assert ft.pop_ast().match(func.body)


def test_syncthreads_not_at_outer_loop():

    @ft.transform
    def test(x0, x1, y):
        x0: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        x1: ft.Var[(4, 5, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 5, 256), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t0 = ft.empty((256,), "int32", "gpu/shared")
            #! label: L1
            for j in range(0, 256):
                t0[j] = x0[i, j]
            for p in range(0, 5):
                t1 = ft.empty((256,), "int32", "gpu/shared")
                #! label: L2
                for j in range(0, 256):
                    t1[j] = x1[i, p, j]
                #! label: L3
                for j in range(0, 256):
                    y[i, p, j] = t0[255 - j] + t1[255 - j]

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    s.parallelize("L3", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    skip_passes=['prop_one_time_use'],
                    verbose=1)

    with ft.VarDef([
        ("x0", (4, 256), "int32", "input", "gpu/global"),
        ("x1", (4, 5, 256), "int32", "input", "gpu/global"),
        ("y", (4, 5, 256), "int32", "output", "gpu/global"),
    ]) as (x0, x1, y):
        with ft.For(".blockIdx.x", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 256) as j:
                with ft.VarDef("t0", (256,), "int32", "cache",
                               "gpu/shared") as t0:
                    ft.Any()  # t0
                    # Not here
                    with ft.For("p", 0, 5) as p:
                        with ft.VarDef("t1", (256,), "int32", "cache",
                                       "gpu/shared") as t1:
                            ft.Any()  # t1
                            ft.Eval(
                                ft.intrinsic(
                                    "__syncthreads()",
                                    has_side_effect=True))  # Here inside p
                            ft.Any()  # L3
                        ft.Eval(
                            ft.intrinsic("__syncthreads()",
                                         has_side_effect=True))
    assert ft.pop_ast().match(func.body)


def test_syncthreads_at_outer_branch_1():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((256,), "int32", "gpu/shared")
            #! label: L1
            for j in range(0, 256):
                t[j] = x[i, j]
            y[i] = t[0] + t[255]

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    verbose=1,
                    skip_passes=['prop_one_time_use'])

    with ft.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For(".blockIdx.x", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 256) as j:
                with ft.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    ft.Any()
                    ft.Eval(
                        ft.intrinsic("__syncthreads()",
                                     has_side_effect=True))  # Here outside If
                    with ft.If(j == 0):
                        ft.Any()
    assert ft.pop_ast().match(func.body)


def test_syncthreads_at_outer_branch_2():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((256,), "int32", "gpu/shared")
            #! label: L1
            for j in range(0, 256):
                t[j] = x[i, j]
            if i < 2:
                y[i] = t[0] + t[255]
            else:
                y[i] = t[0] * t[255]

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    verbose=1,
                    skip_passes=['prop_one_time_use'])

    with ft.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For(".blockIdx.x", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 256) as j:
                with ft.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    ft.Any()
                    ft.Eval(
                        ft.intrinsic("__syncthreads()",
                                     has_side_effect=True))  # Here outside If
                    with ft.If(ft.any()):
                        ft.Any()
                    with ft.Else():
                        ft.Any()
    assert ft.pop_ast().match(func.body)


def test_syncthreads_at_outer_loop_and_outer_branch():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 5, 256), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((256,), "int32", "gpu/shared")
            #! label: L1
            for j in range(0, 256):
                t[j] = x[i, j]
            for p in range(0, 5):
                #! label: L2
                for j in range(0, 256):
                    y[i, p, j] = t[255 - j] + 1

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    skip_passes=['prop_one_time_use'],
                    verbose=1)

    with ft.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4, 5, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For(".blockIdx.x", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 256) as j:
                with ft.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    ft.Any()
                    ft.Eval(
                        ft.intrinsic("__syncthreads()", has_side_effect=True)
                    )  # Here outside p and ouside If
                    with ft.If(j == 0):
                        with ft.For("p", 0, 5) as p:
                            ft.Any()
    assert ft.pop_ast().match(func.body)


def test_syncthreads_between_cond_and_body_of_a_branch():
    # Already normalized AST:
    ast = ft.load_ast('''
@!parallel : @threadIdx.y
for `.threadIdx.y` from 0 until 8 step 1 length 8 {
  @!parallel : @threadIdx.x
  for `.threadIdx.x` from 0 until 32 step 1 length 32 {
    @inout @gpu/global x: float32[8] @!pinned {
      for i from 0 until 5856 step 1 length 5856 {
        if x[`.threadIdx.y`] > 0 {
          if `.threadIdx.x` == 0 {
            x[`.threadIdx.y`] = 1
          }
        }
      }
    }
  }
}
    ''')
    print(ast)

    ast = ft.gpu_make_sync(ast, target)
    print(ast)

    with ft.For("ty", 0, 8) as ty:
        with ft.For("tx", 0, 32) as tx:
            with ft.VarDef("x", (8,), "float32", "inout", "gpu/global") as x:
                with ft.For("i", 0, 5856) as i:
                    with ft.If(x[ty] > 0):
                        with ft.If(tx == 0):
                            x[ty] = 1
                    ft.Eval(
                        ft.intrinsic("__syncwarp(__activemask())",
                                     has_side_effect=True))
    assert ft.pop_ast().match(ast)


def test_syncthreads_split_branch():

    @ft.transform
    def test(x, y, z):
        x: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        z: ft.Var[(4,), "int32", "inout", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((256,), "int32", "gpu/shared")
            #! label: L1
            for j in range(0, 256):
                t[j] = x[i, j]
            z[i] = z[i] + 1
            y[i] = t[0] + t[255]

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    verbose=1,
                    skip_passes=['prop_one_time_use'])

    with ft.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
        ("z", (4,), "int32", "inout", "gpu/global"),
    ]) as (x, y, z):
        with ft.For(".blockIdx.x", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 256) as j:
                with ft.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    ft.Any()
                    with ft.If(j == 0):
                        ft.Any()  # z[i]
                    ft.Eval(
                        ft.intrinsic("__syncthreads()",
                                     has_side_effect=True))  # Here outside If
                    with ft.If(j == 0):
                        ft.Any()  # y[i]
    assert ft.pop_ast().match(func.body)


def test_syncthreads_split_branch_out_of_const_loop():

    @ft.transform
    def test(x, y):
        x: ft.Var[(10, 10, 64), "int32", "input", "gpu/global"]
        y: ft.Var[(10, 10), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(3):
            #! label: L1
            for j in range(4):
                if i * 4 + j < 10:  # THIS BRANCH WILL BE SPLIT
                    #! label: L2
                    for k in range(10):
                        # THIS LOOP WILL BE EXECUTED EVEN WHEN THE OUTER CONDITION
                        # IS EVALUATED TO BE FALSE
                        t = ft.empty((2,), "int32", "gpu/shared")
                        #! label: L3
                        for p in range(64):
                            t[p % 2] += x[i * 4 + j, k, p]
                        y[i * 4 + j, k] = t[0]

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.y")
    s.parallelize("L3", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    with ft.VarDef([("x", (10, 10, 64), "int32", "input", "gpu/global"),
                    ("y", (10, 10), "int32", "output", "gpu/global")]) as (x,
                                                                           y):
        with ft.For(".blockIdx.x", 0, 3) as i:
            with ft.For(".threadIdx.y", 0, 4) as j:
                with ft.For(".threadIdx.x", 0, 64) as p:
                    with ft.For("k", 0, 10) as k:
                        with ft.VarDef("t", (4, 2), "int32", "cache",
                                       "gpu/shared") as t:
                            with ft.If(ft.any()):
                                ft.Any()  # t
                            ft.Eval(
                                ft.intrinsic("__syncthreads()",
                                             has_side_effect=True))
                            with ft.If(ft.any()):
                                with ft.If(p == 0):
                                    ft.Any()  # y
                        ft.Eval(
                            ft.intrinsic("__syncthreads()",
                                         has_side_effect=True))
    assert ft.pop_ast().match(func.body)


def test_syncthreads_no_split_branch_out_of_dynamic_loop():

    @ft.transform
    def test(x, y):
        lim: ft.Var[(3, 4), "int32", "input", "gpu/global"]
        x: ft.Var[(10, 10, 64), "int32", "input", "gpu/global"]
        y: ft.Var[(10, 10), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(3):
            #! label: L1
            for j in range(4):
                if i * 4 + j < 10:  # THIS BRANCH HAS TO BE SPLIT
                    #! label: L2
                    for k in range(lim[i, j]):
                        # THIS LOOP CANNOT BE EXECUTED WHEN THE OUTER CONDITION IS
                        # EVALUATED TO BE FALSE, BEACUSE lim[i, j] MAY BE INVALID
                        t = ft.empty((2,), "int32", "gpu/shared")
                        #! label: L3
                        for p in range(64):
                            t[p % 2] += x[i * 4 + j, k, p]
                        y[i * 4 + j, k] = t[0]

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.y")
    s.parallelize("L3", "threadIdx.x")
    with pytest.raises(ft.InvalidProgram):
        func = ft.lower(s.func(), target, verbose=1)


def test_syncthreads_no_need_to_split_branch():

    @ft.transform
    def test(x, y):
        lim: ft.Var[(3, 4), "int32", "input", "gpu/global"]
        x: ft.Var[(12, 10, 64), "int32", "input", "gpu/global"]
        y: ft.Var[(12, 10), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(3):
            #! label: L1
            for j in range(4):
                if ft.intrinsic("% == %", i, i,
                                ret_type="bool"):  # Whatever condition
                    # NO NEED TO SPLIT THIS BRANCH, BECAUSE IS EVALUATED TO THE
                    # SAME VALUE IN THE SAME THREAD BLOCK, AND WE ONLY SYNC INSIDE
                    # A THREAD BLOCK
                    #! label: L2
                    for k in range(lim[i, j]):
                        t = ft.empty((2,), "int32", "gpu/shared")
                        #! label: L3
                        for p in range(64):
                            t[p % 2] += x[i * 4 + j, k, p]
                        y[i * 4 + j, k] = t[0]

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.y")
    s.parallelize("L3", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    with ft.VarDef([("lim", (3, 4), "int32", "input", "gpu/global"),
                    ("x", (12, 10, 64), "int32", "input", "gpu/global"),
                    ("y", (12, 10), "int32", "output", "gpu/global")
                   ]) as (lim, x, y):
        with ft.For(".blockIdx.x", 0, 3) as i:
            with ft.If(ft.any()):  # HERE OUTSIDE OF k
                with ft.For(".threadIdx.y", 0, 4) as j:
                    with ft.For(".threadIdx.x", 0, 64) as p:
                        with ft.For("k", 0, lim[i, j]) as k:
                            with ft.VarDef("t", (4, 2), "int32", "cache",
                                           "gpu/shared") as t:
                                ft.Any()  # t
                                ft.Eval(
                                    ft.intrinsic("__syncthreads()",
                                                 has_side_effect=True))
                                with ft.If(p == 0):
                                    ft.Any()  # y
                            ft.Eval(
                                ft.intrinsic("__syncthreads()",
                                             has_side_effect=True))
    assert ft.pop_ast().match(func.body)


def test_syncthreads_no_need_to_split_branch_warp():

    @ft.transform
    def test(x, y):
        lim: ft.Var[(3, 4), "int32", "input", "gpu/global"]
        x: ft.Var[(12, 10, 32), "int32", "input", "gpu/global"]
        y: ft.Var[(12, 10), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(3):
            #! label: L1
            for j in range(4):
                if ft.intrinsic("% == %", i, i,
                                ret_type="bool"):  # Whatever condition
                    # NO NEED TO SPLIT THIS BRANCH, BECAUSE WE ONLY NEED
                    # __syncwarp(__activemask()), WHICH IS ACTUALLY A MEMORY FLUSH
                    # INSTEAD OF A SYNCHRONIZATION
                    #! label: L2
                    for k in range(lim[i, j]):
                        t = ft.empty((2,), "int32", "gpu/shared")
                        #! label: L3
                        for p in range(32):
                            t[p % 2] += x[i * 4 + j, k, p]
                        y[i * 4 + j, k] = t[0]

    s = ft.Schedule(test)
    s.parallelize("L0", "threadIdx.z")
    s.parallelize("L1", "threadIdx.y")
    s.parallelize("L3", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    with ft.VarDef([("lim", (3, 4), "int32", "input", "gpu/global"),
                    ("x", (12, 10, 32), "int32", "input", "gpu/global"),
                    ("y", (12, 10), "int32", "output", "gpu/global")
                   ]) as (lim, x, y):
        with ft.For(".threadIdx.z", 0, 3) as i:
            with ft.If(ft.any()):
                with ft.For(".threadIdx.y", 0, 4) as j:
                    with ft.For(".threadIdx.x", 0, 32) as p:
                        with ft.For("k", 0, lim[i, j]) as k:
                            with ft.VarDef("t", (3, 4, 2), "int32", "cache",
                                           "gpu/shared") as t:
                                ft.Any()  # t
                                with ft.If(p == 0):
                                    ft.Eval(
                                        ft.intrinsic(
                                            "__syncwarp(__activemask())",
                                            has_side_effect=True)
                                    )  # HERE INSIDE p == 0
                                    ft.Any()  # y
                            ft.Eval(
                                ft.intrinsic("__syncwarp(__activemask())",
                                             has_side_effect=True))
    assert ft.pop_ast().match(func.body)


def test_syncthreads_split_branch_with_else():

    @ft.transform
    def test(x, y, z):
        x: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        z: ft.Var[(4,), "int32", "inout", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((2,), "int32", "gpu/shared")
            if i < 2:
                #! label: L1
                for j in range(0, 256):
                    t[j % 2] += x[i, j]  # Atomic reduction
                z[i] = z[i] + 1
                y[i] = t[0]
            else:
                #! label: L2
                for j in range(0, 256):
                    t[j % 2] += x[i, j] * 2  # Atomic reduction
                z[i] = z[i] + 1
                y[i] = t[0]

    s = ft.Schedule(test)
    s.parallelize("L0", "threadIdx.y")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    with ft.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
        ("z", (4,), "int32", "inout", "gpu/global"),
    ]) as (x, y, z):
        with ft.For(".threadIdx.y", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 256) as j:
                with ft.VarDef("t", (4, 2), "int32", "cache",
                               "gpu/shared") as t:
                    with ft.If(i < 2):
                        ft.Any()
                        with ft.If(j == 0):
                            ft.Any()  # z[i]
                    ft.Eval(
                        ft.intrinsic("__syncthreads()",
                                     has_side_effect=True))  # Here outside If
                    with ft.If(i < 2):
                        with ft.If(j == 0):
                            ft.Any()  # y[i]

                with ft.VarDef("t", (4, 2), "int32", "cache",
                               "gpu/shared") as t:
                    with ft.If(i >= 2):
                        ft.Any()
                        with ft.If(j == 0):
                            ft.Any()  # z[i]
                    ft.Eval(
                        ft.intrinsic("__syncthreads()",
                                     has_side_effect=True))  # Here outside If
                    with ft.If(i >= 2):
                        with ft.If(j == 0):
                            ft.Any()  # y[i]
    assert ft.pop_ast().match(func.body)


def test_syncthreads_split_branch_but_not_else():

    @ft.transform
    def test(x, y, z):
        x: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        z: ft.Var[(4,), "int32", "inout", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((2,), "int32", "gpu/shared")
            if i < 2:
                # Sync in then case
                #! label: L1
                for j in range(0, 256):
                    t[j % 2] += x[i, j]  # Atomic reduction
                z[i] = z[i] + 1
                y[i] = t[0]
            else:
                # No sync in else case
                z[i] = z[i] + 2

    s = ft.Schedule(test)
    s.parallelize("L0", "threadIdx.y")
    s.parallelize("L1", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    with ft.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
        ("z", (4,), "int32", "inout", "gpu/global"),
    ]) as (x, y, z):
        with ft.For(".threadIdx.y", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 256) as j:
                with ft.VarDef("t", (4, 2), "int32", "cache",
                               "gpu/shared") as t:
                    with ft.If(i < 2):
                        ft.Any()
                        with ft.If(j == 0):
                            ft.Any()  # z[i]
                    ft.Eval(
                        ft.intrinsic("__syncthreads()",
                                     has_side_effect=True))  # Here outside If
                    with ft.If(i < 2):
                        with ft.If(j == 0):
                            ft.Any()  # y[i]
                with ft.If(i >= 2):
                    with ft.If(j == 0):
                        ft.Any()  # z[i]
    assert ft.pop_ast().match(func.body)


def test_syncthreads_split_branch_and_vardef():

    @ft.transform
    def test(x, y, z1, z2):
        x: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        z1: ft.Var[(4,), "int32", "inout", "gpu/global"]
        z2: ft.Var[(4,), "int32", "inout", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((256,), "int32", "gpu/shared")
            #! label: L1
            for j in range(0, 256):
                t[j] = x[i, j]
            u = ft.empty((1,), "int32", "gpu/local")
            u[0] = z1[i] * 2
            y[i] = t[0] + t[255]
            z1[i] = u[0] + 1
            z2[i] = u[0] + 1

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    verbose=1,
                    skip_passes=['prop_one_time_use'])

    with ft.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
        ("z1", (4,), "int32", "inout", "gpu/global"),
        ("z2", (4,), "int32", "inout", "gpu/global"),
    ]) as (x, y, z1, z2):
        with ft.For(".blockIdx.x", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 256) as j:
                with ft.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    ft.Any()
                    with ft.VarDef("u", (1,), "int32", "cache",
                                   "gpu/local") as u:
                        with ft.If(j == 0):
                            ft.Any()  # u[0]
                        ft.Eval(
                            ft.intrinsic(
                                "__syncthreads()",
                                has_side_effect=True))  # Here outside If
                        with ft.If(j == 0):
                            ft.Any()  # y[i]
                            ft.Any()  # z1[i]
                            ft.Any()  # z2[i]
    assert ft.pop_ast().match(func.body)


def test_syncthreads_split_branch_and_vardef_with_else():

    @ft.transform
    def test(x, y, z1, z2):
        x: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(4,), "int32", "output", "gpu/global"]
        z1: ft.Var[(4,), "int32", "inout", "gpu/global"]
        z2: ft.Var[(4,), "int32", "inout", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((2,), "int32", "gpu/shared")
            if i < 2:
                #! label: L1
                for j in range(0, 256):
                    t[j % 2] += x[i, j]  # Atomic reduction
                u1 = ft.empty((1,), "int32", "gpu/local")
                u1[0] = z1[i] * 2
                y[i] = t[0]
                z1[i] = u1[0] + 1
                z2[i] = u1[0] + 1
            else:
                #! label: L2
                for j in range(0, 256):
                    t[j % 2] += x[i, j] * 2  # Atomic reduction
                u2 = ft.empty((1,), "int32", "gpu/local")
                u2[0] = z1[i] * 2
                y[i] = t[0]
                z1[i] = u2[0] + 1
                z2[i] = u2[0] + 1

    s = ft.Schedule(test)
    s.parallelize("L0", "threadIdx.y")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)

    with ft.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
        ("z1", (4,), "int32", "inout", "gpu/global"),
        ("z2", (4,), "int32", "inout", "gpu/global"),
    ]) as (x, y, z1, z2):
        with ft.For(".threadIdx.y", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 256) as j:
                with ft.VarDef("t", (4, 2), "int32", "cache",
                               "gpu/shared") as t:
                    with ft.If(i < 2):
                        ft.Any()
                    with ft.VarDef("u1", (1,), "int32", "cache",
                                   "gpu/local") as u:
                        with ft.If(i < 2):
                            with ft.If(j == 0):
                                ft.Any()  # u[0]
                        ft.Eval(
                            ft.intrinsic(
                                "__syncthreads()",
                                has_side_effect=True))  # Here outside If
                        with ft.If(i < 2):
                            with ft.If(j == 0):
                                ft.Any()  # y[i]
                                ft.Any()  # z1[i]
                                ft.Any()  # z2[i]

                with ft.VarDef("t", (4, 2), "int32", "cache",
                               "gpu/shared") as t:
                    with ft.If(i >= 2):
                        ft.Any()
                    with ft.VarDef("u2", (1,), "int32", "cache",
                                   "gpu/local") as u:
                        with ft.If(i >= 2):
                            with ft.If(j == 0):
                                ft.Any()  # u[0]
                        ft.Eval(
                            ft.intrinsic(
                                "__syncthreads()",
                                has_side_effect=True))  # Here outside If
                        with ft.If(i >= 2):
                            with ft.If(j == 0):
                                ft.Any()  # y[i]
                                ft.Any()  # z1[i]
                                ft.Any()  # z2[i]
    assert ft.pop_ast().match(func.body)


def test_syncwarp():

    @ft.transform
    def test(x, y):
        x: ft.Var[(4, 4), "int32", "input", "gpu/global"]
        y: ft.Var[(4, 4), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            t = ft.empty((4,), "int32", "gpu/shared")
            #! label: L1
            for j in range(0, 4):
                t[j] = x[i, j] * 2
            #! label: L2
            for j in range(0, 4):
                y[i, j] = t[3 - j] + 1

    with ft.VarDef([
        ("x", (4, 4), "int32", "input", "gpu/global"),
        ("y", (4, 4), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, label="L0") as i:
            with ft.VarDef("t", (4,), "int32", "cache", "gpu/shared") as t:
                with ft.For("j1", 0, 4, label="L1") as j:
                    t[j] = x[i, j] * 2
                with ft.For("j2", 0, 4, label="L2") as j:
                    y[i, j] = t[3 - j] + 1
    assert ft.pop_ast().match(test.body)

    s = ft.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ft.lower(s.func(),
                    target,
                    skip_passes=['prop_one_time_use'],
                    verbose=1)

    with ft.VarDef([
        ("x", (4, 4), "int32", "input", "gpu/global"),
        ("y", (4, 4), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For(".blockIdx.x", 0, 4) as i:
            with ft.For(".threadIdx.x", 0, 4) as j:
                with ft.VarDef("t", (4,), "int32", "cache", "gpu/shared") as t:
                    ft.Any()
                    ft.Eval(
                        ft.intrinsic("__syncwarp(__activemask())",
                                     has_side_effect=True))
                    ft.Any()
    assert ft.pop_ast().match(func.body)

    code = ft.codegen(func, target, verbose=True)
    x_np = np.array([[0, 1, 2, 3]] * 4, dtype="int32")
    y_np = np.zeros((4, 4), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([[7, 5, 3, 1]] * 4, dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_use_syncthreads_for_non_aligned_warps():
    with ft.VarDef([("u", (64, 64, 64, 5), "float64", "input", "gpu/global"),
                    ("e", (64, 64, 64, 5), "float64", "output", "gpu/global")
                   ]) as (u, e):
        ft.MarkLabel("L0")
        with ft.For("i", 0, 64) as i:
            ft.MarkLabel("L1")
            with ft.For("j", 0, 64) as j:
                ft.MarkLabel("L2")
                with ft.For("k", 0, 64) as k:
                    with ft.VarDef("t", (5,), "float64", "cache",
                                   "gpu/shared") as t:
                        t[0] = u[i, j, k, 0]
                        t[1] = u[i, j, k, 1]
                        t[2] = u[i, j, k, 2]
                        t[3] = u[i, j, k, 3]
                        t[4] = u[i, j, k, 4]
                        ft.MarkLabel("L3")
                        with ft.For("i1", 0, 5) as i1:
                            e[i, j, k, i1] = t[i1]
    s = ft.Schedule(ft.pop_ast())
    s.parallelize("L0", "blockIdx.y")
    s.parallelize("L1", "blockIdx.x")
    s.parallelize("L2", "threadIdx.y")
    s.parallelize("L3", "threadIdx.x")
    ast = ft.lower(s.ast(), target, verbose=1)

    with ft.VarDef([("u", (64, 64, 64, 5), "float64", "input", "gpu/global"),
                    ("e", (64, 64, 64, 5), "float64", "output", "gpu/global")
                   ]) as (u, e):
        with ft.For("i", 0, 64) as i:
            with ft.For("j", 0, 64) as j:
                with ft.For("k", 0, 64) as k:
                    with ft.For("i1", 0, 5) as i1:
                        with ft.VarDef("t", (64, 5), "float64", "cache",
                                       "gpu/shared") as t:
                            with ft.If(i1 == 0):
                                t[k, 0] = u[i, j, k, 0]
                                t[k, 1] = u[i, j, k, 1]
                                t[k, 2] = u[i, j, k, 2]
                                t[k, 3] = u[i, j, k, 3]
                                t[k, 4] = u[i, j, k, 4]

                            # Use syncthreads here
                            ft.Eval(
                                ft.intrinsic("__syncthreads()",
                                             has_side_effect=True))

                            e[i, j, k, i1] = t[k, i1]
    assert ft.pop_ast().match(ast)


def test_no_sync_between_atomic_reductions():

    @ft.transform
    def test(x, idx1, idx2, y):
        x: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        idx1: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        idx2: ft.Var[(4, 256), "int32", "input", "gpu/global"]
        y: ft.Var[(100,), "int32", "output", "gpu/global"]
        #! label: L0
        for i in range(0, 4):
            #! label: L1
            for j in range(0, 256):
                y[idx1[i, j]] += x[i, j]
                y[idx2[i, j]] += x[i, j]

    s = ft.Schedule(test)
    s.parallelize("L0", "threadIdx.y")
    s.parallelize("L1", "threadIdx.x")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)

    assert "__syncthreads" not in code.code


def test_reject_dependence_between_blocks():
    ast = ft.load_ast('''
@input @gpu/global x: float32[4, 4] {
  @cache @gpu/global t: float32[4, 4] {
    @output @gpu/global y: float32[4, 4] {
      @!parallel : @blockIdx.x
      for i from 0 until 4 step 1 length 4 {
        @!parallel : @blockIdx.y
        for j from 0 until 4 step 1 length 4 {
          t[i, j] = x[i, j] * 2
        }
        @!parallel : @blockIdx.y
        for j_1 from 0 until 4 step 1 length 4 {
          y[i, j_1] = t[i, (j_1 + 1) % 4]
        }
      }
    }
  }
}''')
    with pytest.raises(ft.InvalidProgram):
        ft.lower(ast, target, skip_passes=['prop_one_time_use'])


def test_dont_reject_false_dependence_between_blocks():
    ast = ft.load_ast('''
@input @gpu/global x: float32[4, 4, 4] {
  @cache @gpu/global t: float32[4, 4, 4] {
    @output @gpu/global y: float32[4, 4, 4] {
      for p from 0 until 4 step 1 length 4 {
        @!parallel : @blockIdx.x
        for i from 0 until 4 step 1 length 4 {
          @!parallel : @blockIdx.y
          for j from 0 until 4 step 1 length 4 {
            t[p, i, j] = x[p, i, j] * 2
          }
          if p > 0 {
            @!parallel : @blockIdx.y
            for j_1 from 0 until 4 step 1 length 4 {
              y[p, i, j_1] = t[p - 1, i, (j_1 + 1) % 4]
            }
          }
        }
      }
    }
  }
}''')
    # There shall be no exception
    ft.lower(ast, target, skip_passes=['prop_one_time_use'], verbose=1)
