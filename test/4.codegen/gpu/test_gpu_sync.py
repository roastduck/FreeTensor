import ir
import ir.debug
import pytest
import numpy as np

target = ir.GPU()
device = ir.Device(target)


def test_syncthreads():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 256), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((256,), "int32", "gpu/shared")
            "nid: L1"
            for j in range(0, 256):
                t[j] = x[i, j] * 2
            "nid: L2"
            for j in range(0, 256):
                y[i, j] = t[255 - j] + 1

    with ir.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L0") as i:
            with ir.VarDef("t", (256,), "int32", "cache", "gpu/shared") as t:
                with ir.For("j1", 0, 256, nid="L1") as j:
                    t[j] = x[i, j] * 2
                with ir.For("j2", 0, 256, nid="L2") as j:
                    y[i, j] = t[255 - j] + 1
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For(".blockIdx.x", 0, 4) as i:
            with ir.For(".threadIdx.x", 0, 256) as j:
                with ir.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    ir.Any()
                    ir.Eval(
                        ir.intrinsic("__syncthreads()", has_side_effect=True))
                    ir.Any()
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    x_np = np.array([range(256)] * 4, dtype="int32")
    y_np = np.zeros((4, 256), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([range(511, -1, -2)] * 4, dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_syncthreads_in_loop():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 5, 256), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            for p in range(0, 5):
                t = ir.create_var((256,), "int32", "gpu/shared")
                "nid: L1"
                for j in range(0, 256):
                    t[j] = x[i, j] * p
                "nid: L2"
                for j in range(0, 256):
                    y[i, p, j] = t[255 - j] + 1

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4, 5, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For(".blockIdx.x", 0, 4) as i:
            with ir.For(".threadIdx.x", 0, 256) as j:
                with ir.For("p", 0, 5) as p:
                    with ir.VarDef("t", (256,), "int32", "cache",
                                   "gpu/shared") as t:
                        ir.Any()
                        ir.Eval(
                            ir.intrinsic("__syncthreads()",
                                         has_side_effect=True))
                        ir.Any()
                    ir.Eval(
                        ir.intrinsic("__syncthreads()", has_side_effect=True))
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncthreads_at_outer_loop():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 5, 256), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((256,), "int32", "gpu/shared")
            "nid: L1"
            for j in range(0, 256):
                t[j] = x[i, j]
            for p in range(0, 5):
                "nid: L2"
                for j in range(0, 256):
                    y[i, p, j] = t[255 - j] + 1

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4, 5, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For(".blockIdx.x", 0, 4) as i:
            with ir.For(".threadIdx.x", 0, 256) as j:
                with ir.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    ir.Any()
                    ir.Eval(
                        ir.intrinsic("__syncthreads()",
                                     has_side_effect=True))  # Here outside p
                    with ir.For("p", 0, 5) as p:
                        ir.Any()
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncthreads_not_at_outer_loop():

    @ir.transform
    def test(x0, x1, y):
        ir.declare_var(x0, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(x1, (4, 5, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 5, 256), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t0 = ir.create_var((256,), "int32", "gpu/shared")
            "nid: L1"
            for j in range(0, 256):
                t0[j] = x0[i, j]
            for p in range(0, 5):
                t1 = ir.create_var((256,), "int32", "gpu/shared")
                "nid: L2"
                for j in range(0, 256):
                    t1[j] = x1[i, p, j]
                "nid: L3"
                for j in range(0, 256):
                    y[i, p, j] = t0[255 - j] + t1[255 - j]

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    s.parallelize("L3", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x0", (4, 256), "int32", "input", "gpu/global"),
        ("x1", (4, 5, 256), "int32", "input", "gpu/global"),
        ("y", (4, 5, 256), "int32", "output", "gpu/global"),
    ]) as (x0, x1, y):
        with ir.For(".blockIdx.x", 0, 4) as i:
            with ir.For(".threadIdx.x", 0, 256) as j:
                with ir.VarDef("t0", (256,), "int32", "cache",
                               "gpu/shared") as t0:
                    ir.Any()  # t0
                    # Not here
                    with ir.For("p", 0, 5) as p:
                        with ir.VarDef("t1", (256,), "int32", "cache",
                                       "gpu/shared") as t1:
                            ir.Any()  # t1
                            ir.Eval(
                                ir.intrinsic(
                                    "__syncthreads()",
                                    has_side_effect=True))  # Here inside p
                            ir.Any()  # L3
                        ir.Eval(
                            ir.intrinsic("__syncthreads()",
                                         has_side_effect=True))
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncthreads_at_outer_branch():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((256,), "int32", "gpu/shared")
            "nid: L1"
            for j in range(0, 256):
                t[j] = x[i, j]
            y[i] = t[0] + t[1]

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For(".blockIdx.x", 0, 4) as i:
            with ir.For(".threadIdx.x", 0, 256) as j:
                with ir.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    ir.Any()
                    ir.Eval(
                        ir.intrinsic("__syncthreads()",
                                     has_side_effect=True))  # Here outside If
                    with ir.If(j == 0):
                        ir.Any()
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncthreads_at_outer_loop_and_outer_branch():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 5, 256), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((256,), "int32", "gpu/shared")
            "nid: L1"
            for j in range(0, 256):
                t[j] = x[i, j]
            for p in range(0, 5):
                "nid: L2"
                for j in range(0, 256):
                    y[i, p, j] = t[255 - j] + 1

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4, 5, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For(".blockIdx.x", 0, 4) as i:
            with ir.For(".threadIdx.x", 0, 256) as j:
                with ir.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    ir.Any()
                    ir.Eval(
                        ir.intrinsic("__syncthreads()", has_side_effect=True)
                    )  # Here outside p and ouside If
                    with ir.If(j == 0):
                        with ir.For("p", 0, 5) as p:
                            ir.Any()
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncthreads_split_branch():

    @ir.transform
    def test(x, y, z):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        ir.declare_var(z, (4,), "int32", "inout", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((256,), "int32", "gpu/shared")
            "nid: L1"
            for j in range(0, 256):
                t[j] = x[i, j]
            z[i] = z[i] + 1
            y[i] = t[0] + t[1]

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
        ("z", (4,), "int32", "inout", "gpu/global"),
    ]) as (x, y, z):
        with ir.For(".blockIdx.x", 0, 4) as i:
            with ir.For(".threadIdx.x", 0, 256) as j:
                with ir.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    ir.Any()
                    with ir.If(j == 0):
                        ir.Any()  # z[i]
                    ir.Eval(
                        ir.intrinsic("__syncthreads()",
                                     has_side_effect=True))  # Here outside If
                    with ir.If(j == 0):
                        ir.Any()  # y[i]
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncthreads_split_branch_out_of_const_loop():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (10, 10, 32), "int32", "input", "gpu/global")
        ir.declare_var(y, (10, 10), "int32", "output", "gpu/global")
        'nid: L0'
        for i in range(3):
            'nid: L1'
            for j in range(4):
                if i * 4 + j < 10:
                    'nid: L2'
                    for k in range(10):
                        t = ir.create_var((2,), "int32", "gpu/shared")
                        'nid: L3'
                        for p in range(32):
                            t[p % 2] += x[i * 4 + j, k, p]
                        y[i * 4 + j, k] = t[0]

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.y")
    s.parallelize("L3", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([("x", (10 * 10 * 32,), "int32", "input", "gpu/global"),
                    ("y", (10 * 10,), "int32", "output", "gpu/global")]) as (x,
                                                                             y):
        with ir.For(".blockIdx.x", 0, 3) as i:
            with ir.For(".threadIdx.y", 0, 4) as j:
                with ir.For(".threadIdx.x", 0, 32) as p:
                    with ir.For("k", 0, 10) as k:
                        with ir.VarDef("t", (4 * 2,), "int32", "cache",
                                       "gpu/shared") as t:
                            with ir.If(ir.any()):
                                ir.Any()  # t
                            ir.Eval(
                                ir.intrinsic("__syncwarp()",
                                             has_side_effect=True))
                            with ir.If(ir.any()):
                                with ir.If(p == 0):
                                    ir.Any()  # y
                        ir.Eval(
                            ir.intrinsic("__syncwarp()", has_side_effect=True))
    assert ir.pop_ast().match(func.body)


def test_syncthreads_split_branch_with_else():

    @ir.transform
    def test(x, y, z):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        ir.declare_var(z, (4,), "int32", "inout", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((2,), "int32", "gpu/shared")
            if i < 2:
                "nid: L1"
                for j in range(0, 256):
                    t[j % 2] += x[i, j]  # Atomic reduction
                z[i] = z[i] + 1
                y[i] = t[0]
            else:
                "nid: L2"
                for j in range(0, 256):
                    t[j % 2] += x[i, j] * 2  # Atomic reduction
                z[i] = z[i] + 1
                y[i] = t[0]

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
        ("z", (4,), "int32", "inout", "gpu/global"),
    ]) as (x, y, z):
        with ir.For(".blockIdx.x", 0, 4) as i:
            with ir.For(".threadIdx.x", 0, 256) as j:
                with ir.VarDef("t", (2,), "int32", "cache", "gpu/shared") as t:
                    with ir.If(i < 2):
                        ir.Any()
                        with ir.If(j == 0):
                            ir.Any()  # z[i]
                    ir.Eval(
                        ir.intrinsic("__syncthreads()",
                                     has_side_effect=True))  # Here outside If
                    with ir.If(i < 2):
                        with ir.If(j == 0):
                            ir.Any()  # y[i]

                with ir.VarDef("t", (2,), "int32", "cache", "gpu/shared") as t:
                    with ir.If(i >= 2):
                        ir.Any()
                        with ir.If(j == 0):
                            ir.Any()  # z[i]
                    ir.Eval(
                        ir.intrinsic("__syncthreads()",
                                     has_side_effect=True))  # Here outside If
                    with ir.If(i >= 2):
                        with ir.If(j == 0):
                            ir.Any()  # y[i]
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncthreads_split_branch_and_vardef():

    @ir.transform
    def test(x, y, z1, z2):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        ir.declare_var(z1, (4,), "int32", "inout", "gpu/global")
        ir.declare_var(z2, (4,), "int32", "inout", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((256,), "int32", "gpu/shared")
            "nid: L1"
            for j in range(0, 256):
                t[j] = x[i, j]
            u = ir.create_var((1,), "int32", "gpu/local")
            u[0] = z1[i] * 2
            y[i] = t[0] + t[1]
            z1[i] = u[0] + 1
            z2[i] = u[0] + 1

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
        ("z1", (4,), "int32", "inout", "gpu/global"),
        ("z2", (4,), "int32", "inout", "gpu/global"),
    ]) as (x, y, z1, z2):
        with ir.For(".blockIdx.x", 0, 4) as i:
            with ir.For(".threadIdx.x", 0, 256) as j:
                with ir.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    ir.Any()
                    with ir.VarDef("u", (1,), "int32", "cache",
                                   "gpu/shared") as u:
                        with ir.If(j == 0):
                            ir.Any()  # u[0]
                        ir.Eval(
                            ir.intrinsic(
                                "__syncthreads()",
                                has_side_effect=True))  # Here outside If
                        with ir.If(j == 0):
                            ir.Any()  # y[i]
                            ir.Any()  # z1[i]
                            ir.Any()  # z2[i]
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncthreads_split_branch_and_vardef_with_else():

    @ir.transform
    def test(x, y, z1, z2):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        ir.declare_var(z1, (4,), "int32", "inout", "gpu/global")
        ir.declare_var(z2, (4,), "int32", "inout", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((2,), "int32", "gpu/shared")
            if i < 2:
                "nid: L1"
                for j in range(0, 256):
                    t[j % 2] += x[i, j]  # Atomic reduction
                u1 = ir.create_var((1,), "int32", "gpu/local")
                u1[0] = z1[i] * 2
                y[i] = t[0]
                z1[i] = u1[0] + 1
                z2[i] = u1[0] + 1
            else:
                "nid: L2"
                for j in range(0, 256):
                    t[j % 2] += x[i, j] * 2  # Atomic reduction
                u2 = ir.create_var((1,), "int32", "gpu/local")
                u2[0] = z1[i] * 2
                y[i] = t[0]
                z1[i] = u2[0] + 1
                z2[i] = u2[0] + 1

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
        ("z1", (4,), "int32", "inout", "gpu/global"),
        ("z2", (4,), "int32", "inout", "gpu/global"),
    ]) as (x, y, z1, z2):
        with ir.For(".blockIdx.x", 0, 4) as i:
            with ir.For(".threadIdx.x", 0, 256) as j:
                with ir.VarDef("t", (2,), "int32", "cache", "gpu/shared") as t:
                    with ir.If(i < 2):
                        ir.Any()
                    with ir.VarDef("u1", (1,), "int32", "cache",
                                   "gpu/shared") as u:
                        with ir.If(i < 2):
                            with ir.If(j == 0):
                                ir.Any()  # u[0]
                        ir.Eval(
                            ir.intrinsic(
                                "__syncthreads()",
                                has_side_effect=True))  # Here outside If
                        with ir.If(i < 2):
                            with ir.If(j == 0):
                                ir.Any()  # y[i]
                                ir.Any()  # z1[i]
                                ir.Any()  # z2[i]

                with ir.VarDef("t", (2,), "int32", "cache", "gpu/shared") as t:
                    with ir.If(i >= 2):
                        ir.Any()
                    with ir.VarDef("u2", (1,), "int32", "cache",
                                   "gpu/shared") as u:
                        with ir.If(i >= 2):
                            with ir.If(j == 0):
                                ir.Any()  # u[0]
                        ir.Eval(
                            ir.intrinsic(
                                "__syncthreads()",
                                has_side_effect=True))  # Here outside If
                        with ir.If(i >= 2):
                            with ir.If(j == 0):
                                ir.Any()  # y[i]
                                ir.Any()  # z1[i]
                                ir.Any()  # z2[i]
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncwarp():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 4), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 4), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((4,), "int32", "gpu/shared")
            "nid: L1"
            for j in range(0, 4):
                t[j] = x[i, j] * 2
            "nid: L2"
            for j in range(0, 4):
                y[i, j] = t[3 - j] + 1

    with ir.VarDef([
        ("x", (4, 4), "int32", "input", "gpu/global"),
        ("y", (4, 4), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L0") as i:
            with ir.VarDef("t", (4,), "int32", "cache", "gpu/shared") as t:
                with ir.For("j1", 0, 4, nid="L1") as j:
                    t[j] = x[i, j] * 2
                with ir.For("j2", 0, 4, nid="L2") as j:
                    y[i, j] = t[3 - j] + 1
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (4, 4), "int32", "input", "gpu/global"),
        ("y", (4, 4), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For(".blockIdx.x", 0, 4) as i:
            with ir.For(".threadIdx.x", 0, 4) as j:
                with ir.VarDef("t", (4,), "int32", "cache", "gpu/shared") as t:
                    ir.Any()
                    ir.Eval(ir.intrinsic("__syncwarp()", has_side_effect=True))
                    ir.Any()
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    x_np = np.array([[0, 1, 2, 3]] * 4, dtype="int32")
    y_np = np.zeros((4, 4), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([[7, 5, 3, 1]] * 4, dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_use_syncthreads_for_non_aligned_warps():
    with ir.VarDef([("u", (64, 64, 64, 5), "float64", "input", "gpu/global"),
                    ("e", (64, 64, 64, 5), "float64", "output", "gpu/global")
                   ]) as (u, e):
        ir.MarkNid("L0")
        with ir.For("i", 0, 64) as i:
            ir.MarkNid("L1")
            with ir.For("j", 0, 64) as j:
                ir.MarkNid("L2")
                with ir.For("k", 0, 64) as k:
                    with ir.VarDef("t", (5,), "float64", "cache",
                                   "gpu/shared") as t:
                        t[0] = u[i, j, k, 0]
                        t[1] = u[i, j, k, 1]
                        t[2] = u[i, j, k, 2]
                        t[3] = u[i, j, k, 3]
                        t[4] = u[i, j, k, 4]
                        ir.MarkNid("L3")
                        with ir.For("i1", 0, 5) as i1:
                            e[i, j, k, i1] = t[i1]
    s = ir.Schedule(ir.pop_ast())
    s.parallelize("L0", "blockIdx.y")
    s.parallelize("L1", "blockIdx.x")
    s.parallelize("L2", "threadIdx.y")
    s.parallelize("L3", "threadIdx.x")
    ast = ir.lower(s.ast(), target)
    print(ast)

    with ir.VarDef([("u", (64, 64, 64, 5), "float64", "input", "gpu/global"),
                    ("e", (64, 64, 64, 5), "float64", "output", "gpu/global")
                   ]) as (u, e):
        with ir.For("i", 0, 64) as i:
            with ir.For("j", 0, 64) as j:
                with ir.For("k", 0, 64) as k:
                    with ir.For("i1", 0, 5) as i1:
                        with ir.VarDef("t", (64, 5), "float64", "cache",
                                       "gpu/shared") as t:
                            with ir.If(i1 == 0):
                                t[k, 0] = u[i, j, k, 0]
                                t[k, 1] = u[i, j, k, 1]
                                t[k, 2] = u[i, j, k, 2]
                                t[k, 3] = u[i, j, k, 3]
                                t[k, 4] = u[i, j, k, 4]

                            # Use syncthreads here
                            ir.Eval(
                                ir.intrinsic("__syncthreads()",
                                             has_side_effect=True))

                            e[i, j, k, i1] = t[k, i1]
    assert ir.make_1d_var(ir.pop_ast()).match(ast)
