import ir
import ir.debug
import pytest
import numpy as np

target = ir.GPU()
device = ir.Device(target)

host = ir.Device(ir.CPU())


def test_basic():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4,), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            y[i] = x[i] + 1

    with ir.VarDef([
        ("x", (4,), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            y[i] = x[i] + 1
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L1", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_split_by_block_and_bind():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (100,), "int32", "input", "gpu/global")
        ir.declare_var(y, (100,), "int32", "output", "gpu/global")
        "nid: L1"
        for i in range(0, 100):
            y[i] = x[i] + 1

    s = ir.Schedule(test)
    outer, inner = s.split("L1", nparts=3)
    s.parallelize(outer, "blockIdx.x")
    s.parallelize(inner, "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (100,), "int32", "input", "gpu/global"),
        ("y", (100,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For(".blockIdx.x", 0, 3) as i:
            with ir.For(".threadIdx.x", 0, 34) as j:
                with ir.If(ir.any()):
                    ir.Any()
    assert ir.pop_ast().match(func.body)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    x_np = np.array(range(0, 100), dtype="int32")
    y_np = np.zeros((100,), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.array(range(1, 101), dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_shmem():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4,), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            "nid: S1"
            y[i] = x[i] + 1

    with ir.VarDef([
        ("x", (4,), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            ir.MarkNid("S1")
            y[i] = x[i] + 1
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.cache("S1", "x", "gpu/shared")
    s.parallelize("L1", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    assert "__shared__" in code
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_global_mem():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4,), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        t = ir.create_var((4,), "int32", "cache", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            t[i] = x[i] * 2
        "nid: L2"
        for i in range(0, 4):
            y[i] = t[i] + 1

    with ir.VarDef([
        ("x", (4,), "int32", "input", "gpu/global"),
        ("y", (4,), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.VarDef("t", (4,), "int32", "cache", "gpu/global") as t:
            with ir.For("i1", 0, 4, nid="L1") as i:
                t[i] = x[i] * 2
            with ir.For("i2", 0, 4, nid="L2") as i:
                y[i] = t[i] + 1
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    assert "cudaMalloc" in code
    assert "cudaFree" in code
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.array([3, 5, 7, 9], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_pass_by_value_0d():

    @ir.transform
    def test(n, x, y):
        ir.declare_var(n, (), "int32", "input", "byvalue")
        ir.declare_var(x, (n[()], 4), "int32", "input", "gpu/global")
        ir.declare_var(y, (n[()], 4), "int32", "output", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            "nid: L2"
            for j in range(0, n[()]):
                y[j, i] = x[j, i] + 1

    with ir.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ir.VarDef([
            ("x", (n[()], 4), "int32", "input", "gpu/global"),
            ("y", (n[()], 4), "int32", "output", "gpu/global"),
        ]) as (x, y):
            with ir.For("i", 0, 4, nid="L1") as i:
                with ir.For("j", 0, n[()], nid="L2") as j:
                    y[j, i] = x[j, i] + 1
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L1", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    n_np = np.array(5, dtype="int32")
    x_np = np.array([[1, 2, 3, 4]] * 5, dtype="int32")
    y_np = np.zeros((5, 4), dtype="int32")
    n_arr = ir.Array(n_np, host)
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(n=n_arr, x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(5, 4)

    y_std = np.array([[2, 3, 4, 5]] * 5, dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_pass_by_value_1d():

    @ir.transform
    def test(n, x, y):
        ir.declare_var(n, (1,), "int32", "input", "byvalue")
        ir.declare_var(x, (n[0], 4), "int32", "input", "gpu/global")
        ir.declare_var(y, (n[0], 4), "int32", "output", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            "nid: L2"
            for j in range(0, n[0]):
                y[j, i] = x[j, i] + 1

    with ir.VarDef("n", (1,), "int32", "input", "byvalue") as n:
        with ir.VarDef([
            ("x", (n[0], 4), "int32", "input", "gpu/global"),
            ("y", (n[0], 4), "int32", "output", "gpu/global"),
        ]) as (x, y):
            with ir.For("i", 0, 4, nid="L1") as i:
                with ir.For("j", 0, n[0], nid="L2") as j:
                    y[j, i] = x[j, i] + 1
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L1", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    n_np = np.array([5], dtype="int32")
    x_np = np.array([[1, 2, 3, 4]] * 5, dtype="int32")
    y_np = np.zeros((5, 4), dtype="int32")
    n_arr = ir.Array(n_np, host)
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(n=n_arr, x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(5, 4)

    y_std = np.array([[2, 3, 4, 5]] * 5, dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_dynamic_2d_array():
    with ir.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ir.VarDef([
            ("x", (n[()], n[()]), "int32", "input", "gpu/global"),
            ("y", (n[()], n[()]), "int32", "output", "gpu/global"),
        ]) as (x, y):
            with ir.For("i", 0, n[()], nid="L1") as i:
                with ir.For("j", 0, n[()], nid="L2") as j:
                    y[i, j] = x[i, j] + 1

    s = ir.Schedule(ir.Func("main", ["n", "x", "y"], ir.pop_ast()))
    outer, inner = s.split("L1", 4)
    s.reorder([inner, outer])
    s.parallelize(inner, "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    n_np = np.array(5, dtype="int32")
    x_np = np.random.randint(0, 100, (5, 5)).astype("int32")
    y_np = np.zeros((5, 5), dtype="int32")
    n_arr = ir.Array(n_np, host)
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(n=n_arr, x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(5, 5)

    y_std = x_np + 1
    assert np.array_equal(y_np, y_std)


def test_intrinsic():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4,), "float32", "input", "gpu/global")
        ir.declare_var(y, (4,), "float32", "output", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            y[i] = ir.intrinsic("sinf(%)", x[i], ret_type="float32")

    with ir.VarDef([
        ("x", (4,), "float32", "input", "gpu/global"),
        ("y", (4,), "float32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            y[i] = ir.intrinsic("sinf(%)", x[i], ret_type="float32")
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L1", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    x_np = np.array([1, 2, 3, 4], dtype="float32")
    y_np = np.zeros((4,), dtype="float32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.array(np.sin(x_np), dtype="float32")
    assert np.all(np.isclose(y_np, y_std))


def test_syncthreads():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 256), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((256,), "int32", "cache", "gpu/shared")
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
                    ir.Eval(ir.intrinsic("__syncthreads()"))
                    ir.Any()
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    x_np = np.array([range(256)] * 4, dtype="int32")
    y_np = np.zeros((4, 256), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(4, 256)

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
                t = ir.create_var((256,), "int32", "cache", "gpu/shared")
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
                        ir.Eval(ir.intrinsic("__syncthreads()"))
                        ir.Any()
                    ir.Eval(ir.intrinsic("__syncthreads()"))
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncthreads_at_outer_loop():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 5, 256), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((256,), "int32", "cache", "gpu/shared")
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
                    ir.Eval(ir.intrinsic("__syncthreads()"))  # Here outside p
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
            t0 = ir.create_var((256,), "int32", "cache", "gpu/shared")
            "nid: L1"
            for j in range(0, 256):
                t0[j] = x0[i, j]
            for p in range(0, 5):
                t1 = ir.create_var((256,), "int32", "cache", "gpu/shared")
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
                            ir.Eval(ir.intrinsic(
                                "__syncthreads()"))  # Here inside p
                            ir.Any()  # L3
                        ir.Eval(ir.intrinsic("__syncthreads()"))
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncthreads_at_outer_branch():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((1,), "int32", "cache", "gpu/shared")
            "nid: L1"
            for j in range(0, 256):
                t[0] = t[0] + x[i, j]  # Atomic reduction
            y[i] = t[0]

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
                with ir.VarDef("t", (1,), "int32", "cache", "gpu/shared") as t:
                    ir.Any()
                    ir.Eval(ir.intrinsic("__syncthreads()"))  # Here outside If
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
            t = ir.create_var((256,), "int32", "cache", "gpu/shared")
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
                    ir.Eval(ir.intrinsic(
                        "__syncthreads()"))  # Here outside p and ouside If
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
            t = ir.create_var((1,), "int32", "cache", "gpu/shared")
            "nid: L1"
            for j in range(0, 256):
                t[0] = t[0] + x[i, j]  # Atomic reduction
            z[i] = z[i] + 1
            y[i] = t[0]

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
                with ir.VarDef("t", (1,), "int32", "cache", "gpu/shared") as t:
                    ir.Any()
                    with ir.If(j == 0):
                        ir.Any()  # z[i]
                    ir.Eval(ir.intrinsic("__syncthreads()"))  # Here outside If
                    with ir.If(j == 0):
                        ir.Any()  # y[i]
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncthreads_split_branch_with_else():

    @ir.transform
    def test(x, y, z):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        ir.declare_var(z, (4,), "int32", "inout", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((1,), "int32", "cache", "gpu/shared")
            if i < 2:
                "nid: L1"
                for j in range(0, 256):
                    t[0] = t[0] + x[i, j]  # Atomic reduction
                z[i] = z[i] + 1
                y[i] = t[0]
            else:
                "nid: L2"
                for j in range(0, 256):
                    t[0] = t[0] + x[i, j] * 2  # Atomic reduction
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
                with ir.VarDef("t", (1,), "int32", "cache", "gpu/shared") as t:
                    with ir.If(i < 2):
                        ir.Any()
                        with ir.If(j == 0):
                            ir.Any()  # z[i]
                    ir.Eval(ir.intrinsic("__syncthreads()"))  # Here outside If
                    with ir.If(i < 2):
                        with ir.If(j == 0):
                            ir.Any()  # y[i]
                    with ir.If(i >= 2):
                        ir.Any()
                        with ir.If(j == 0):
                            ir.Any()  # z[i]
                    ir.Eval(ir.intrinsic("__syncthreads()"))  # Here outside If
                    with ir.If(i >= 2):
                        with ir.If(j == 0):
                            ir.Any()  # y[i]
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncthreads_split_branch_and_vardef():

    @ir.transform
    def test(x, y, z):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        ir.declare_var(z, (4,), "int32", "inout", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((1,), "int32", "cache", "gpu/shared")
            "nid: L1"
            for j in range(0, 256):
                t[0] = t[0] + x[i, j]  # Atomic reduction
            u = ir.create_var((1,), "int32", "cache", "gpu/local")
            u[0] = z[i] * 2
            y[i] = t[0]
            z[i] = u[0] + 1

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
                with ir.VarDef("t", (1,), "int32", "cache", "gpu/shared") as t:
                    ir.Any()
                    with ir.VarDef("u", (1,), "int32", "cache",
                                   "gpu/shared") as u:
                        with ir.If(j == 0):
                            ir.Any()  # u[0]
                        ir.Eval(
                            ir.intrinsic("__syncthreads()"))  # Here outside If
                        with ir.If(j == 0):
                            ir.Any()  # y[i]
                            ir.Any()  # z[i]
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncthreads_split_branch_and_vardef_with_else():

    @ir.transform
    def test(x, y, z):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        ir.declare_var(z, (4,), "int32", "inout", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((1,), "int32", "cache", "gpu/shared")
            if i < 2:
                "nid: L1"
                for j in range(0, 256):
                    t[0] = t[0] + x[i, j]  # Atomic reduction
                u1 = ir.create_var((1,), "int32", "cache", "gpu/local")
                u1[0] = z[i] * 2
                y[i] = t[0]
                z[i] = u1[0] + 1
            else:
                "nid: L2"
                for j in range(0, 256):
                    t[0] = t[0] + x[i, j] * 2  # Atomic reduction
                u2 = ir.create_var((1,), "int32", "cache", "gpu/local")
                u2[0] = z[i] * 2
                y[i] = t[0]
                z[i] = u2[0] + 1

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
                with ir.VarDef("t", (1,), "int32", "cache", "gpu/shared") as t:
                    with ir.VarDef("u1", (1,), "int32", "cache",
                                   "gpu/shared") as u:
                        with ir.If(i < 2):
                            ir.Any()
                            with ir.If(j == 0):
                                ir.Any()  # u[0]
                        ir.Eval(
                            ir.intrinsic("__syncthreads()"))  # Here outside If
                        with ir.If(i < 2):
                            with ir.If(j == 0):
                                ir.Any()  # y[i]
                                ir.Any()  # z[i]
                    with ir.VarDef("u2", (1,), "int32", "cache",
                                   "gpu/shared") as u:
                        with ir.If(i >= 2):
                            ir.Any()
                            with ir.If(j == 0):
                                ir.Any()  # u[0]
                        ir.Eval(
                            ir.intrinsic("__syncthreads()"))  # Here outside If
                        with ir.If(i >= 2):
                            with ir.If(j == 0):
                                ir.Any()  # y[i]
                                ir.Any()  # z[i]
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_syncwarp():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 4), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 4), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((4,), "int32", "cache", "gpu/shared")
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
                    ir.Eval(ir.intrinsic("__syncwarp()"))
                    ir.Any()
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    x_np = np.array([[0, 1, 2, 3]] * 4, dtype="int32")
    y_np = np.zeros((4, 4), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(4, 4)

    y_std = np.array([[7, 5, 3, 1]] * 4, dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_correct_shared():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 256), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((256,), "int32", "cache", "gpu/shared")
            "nid: L1"
            for j in range(0, 256):
                t[j] = x[i, j] * 2
            "nid: L2"
            for j in range(0, 256):
                y[i, j] = t[j] + 1

    with ir.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L0") as i:
            with ir.VarDef("t", (256,), "int32", "cache", "gpu/shared") as t:
                with ir.For("j1", 0, 256, nid="L1") as j:
                    t[j] = x[i, j] * 2
                with ir.For("j2", 0, 256, nid="L2") as j:
                    y[i, j] = t[j] + 1
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L0", "threadIdx.y")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (4, 256), "int32", "input", "gpu/global"),
        ("y", (4, 256), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For(".threadIdx.y", 0, 4) as i:
            with ir.For(".threadIdx.x", 0, 256) as j:
                with ir.VarDef("t", (4, 256), "int32", "cache",
                               "gpu/shared") as t:
                    t[i, j] = x[i, j] * 2
                    y[i, j] = t[i, j] + 1
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    x_np = np.array([range(256)] * 4, dtype="int32")
    y_np = np.zeros((4, 256), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(4, 256)

    y_std = np.array([range(1, 513, 2)] * 4, dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_relax_shared_shape_to_constants():
    with ir.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ir.VarDef([
            ("x", (4, 256), "int32", "input", "gpu/global"),
            ("y", (4, 256), "int32", "output", "gpu/global"),
        ]) as (x, y):
            with ir.Assert(n[()] <= 256):
                with ir.For("i", 0, 4, nid="L0") as i:
                    with ir.VarDef("t", (n[()],), "int32", "cache",
                                   "gpu/shared") as t:
                        with ir.For("j", 0, n[()], nid="L1") as j:
                            t[j] = x[i, j] * 2
                        with ir.For("j", 0, n[()], nid="L2") as j:
                            y[i, j] = t[j] + 1
                        with ir.For("j", n[()], 256, nid="L3") as j:
                            y[i, j] = 0

    s = ir.Schedule(ir.Func("main", ["n", "x", "y"], ir.pop_ast()))
    s.parallelize("L0", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ir.VarDef([
            ("x", (4, 256), "int32", "input", "gpu/global"),
            ("y", (4, 256), "int32", "output", "gpu/global"),
        ]) as (x, y):
            with ir.Assert(n[()] <= 256):
                with ir.For(".threadIdx.y", 0, 4) as i:
                    with ir.VarDef("t", (4, 256), "int32", "cache",
                                   "gpu/shared") as t:
                        with ir.For("j", 0, n[()]) as j:
                            t[i, j] = x[i, j] * 2
                        with ir.For("j", 0, n[()]) as j:
                            y[i, j] = t[i, j] + 1
                    with ir.For("j", n[()], 256) as j:
                        y[i, j] = 0
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    n_np = np.array(200, dtype="int32")
    x_np = np.array([range(256)] * 4, dtype="int32")
    y_np = np.zeros((4, 256), dtype="int32")
    n_arr = ir.Array(n_np, host)
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(n=n_arr, x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(4, 256)

    y_std = np.array([list(range(1, 401, 2)) + [0] * 56] * 4, dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_parallel_different_length():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (4, 4), "int32", "input", "gpu/global")
        ir.declare_var(b, (4, 8), "int32", "input", "gpu/global")
        ir.declare_var(c, (4, 8), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((4,), "int32", "cache", "gpu/shared")
            "nid: L1"
            for j in range(0, 4):
                t[j] = a[i, j]
            "nid: L2"
            for j in range(0, 4):
                "nid: L3"
                for k in range(0, 8):
                    c[i, k] = c[i, k] + t[j] * b[j, k]

    with ir.VarDef([
        ("a", (4, 4), "int32", "input", "gpu/global"),
        ("b", (4, 8), "int32", "input", "gpu/global"),
        ("c", (4, 8), "int32", "output", "gpu/global"),
    ]) as (a, b, c):
        with ir.For("i", 0, 4, nid="L0") as i:
            with ir.VarDef("t", (4,), "int32", "cache", "gpu/shared") as t:
                with ir.For("j1", 0, 4, nid="L1") as j:
                    t[j] = a[i, j]
                with ir.For("j2", 0, 4, nid="L2") as j:
                    with ir.For("k", 0, 8, nid="L3") as k:
                        c[i, k] = c[i, k] + t[j] * b[j, k]
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L3", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("a", (4, 4), "int32", "input", "gpu/global"),
        ("b", (4, 8), "int32", "input", "gpu/global"),
        ("c", (4, 8), "int32", "output", "gpu/global"),
    ]) as (a, b, c):
        with ir.For(".blockIdx.x", 0, 4) as blk:
            with ir.For(".threadIdx.x", 0, 8) as th:
                with ir.VarDef("t", (4,), "int32", "cache", "gpu/shared") as t:
                    with ir.If(th < 4):
                        t[th] = a[blk, th]
                    ir.Eval(ir.intrinsic("__syncwarp()"))
                    with ir.For("j", 0, 4) as j:
                        ir.Any()
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    a_np = np.random.randint(0, 100, (4, 4)).astype("int32")
    b_np = np.random.randint(0, 100, (4, 8)).astype("int32")
    c_np = np.zeros((4, 8), dtype="int32")
    a_arr = ir.Array(a_np, device)
    b_arr = ir.Array(b_np, device)
    c_arr = ir.Array(c_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(a=a_arr, b=b_arr, c=c_arr)
    driver.run()
    c_np = c_arr.numpy().reshape(4, 8)

    c_std = a_np @ b_np
    assert np.array_equal(c_np, c_std)


def test_parallel_broadcast():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (4, 1), "int32", "input", "gpu/global")
        ir.declare_var(b, (1, 8), "int32", "input", "gpu/global")
        ir.declare_var(c, (4, 8), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((1,), "int32", "cache", "gpu/shared")
            t[0] = a[i, 0]
            "nid: L1"
            for k in range(0, 8):
                c[i, k] = c[i, k] + t[0] * b[0, k]

    with ir.VarDef([
        ("a", (4, 1), "int32", "input", "gpu/global"),
        ("b", (1, 8), "int32", "input", "gpu/global"),
        ("c", (4, 8), "int32", "output", "gpu/global"),
    ]) as (a, b, c):
        with ir.For("i", 0, 4, nid="L0") as i:
            with ir.VarDef("t", (1,), "int32", "cache", "gpu/shared") as t:
                t[0] = a[i, 0]
                with ir.For("k", 0, 8, nid="L1") as k:
                    c[i, k] = c[i, k] + t[0] * b[0, k]
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L0", "blockIdx.x")
    s.parallelize("L1", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("a", (4, 1), "int32", "input", "gpu/global"),
        ("b", (1, 8), "int32", "input", "gpu/global"),
        ("c", (4, 8), "int32", "output", "gpu/global"),
    ]) as (a, b, c):
        with ir.For(".blockIdx.x", 0, 4) as blk:
            with ir.For(".threadIdx.x", 0, 8) as th:
                with ir.VarDef("t", (1,), "int32", "cache", "gpu/shared") as t:
                    with ir.If(th == 0):
                        t[0] = a[blk, 0]
                    ir.Eval(ir.intrinsic("__syncwarp()"))
                    ir.Any()
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    a_np = np.random.randint(0, 100, (4, 1)).astype("int32")
    b_np = np.random.randint(0, 100, (1, 8)).astype("int32")
    c_np = np.zeros((4, 8), dtype="int32")
    a_arr = ir.Array(a_np, device)
    b_arr = ir.Array(b_np, device)
    c_arr = ir.Array(c_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(a=a_arr, b=b_arr, c=c_arr)
    driver.run()
    c_np = c_arr.numpy().reshape(4, 8)

    c_std = a_np @ b_np
    assert np.array_equal(c_np, c_std)


def test_unbounded_length():

    @ir.transform
    def test(n, x, y):
        ir.declare_var(n, (), "int32", "input", "gpu/global")
        ir.declare_var(x, (n[()],), "int32", "input", "gpu/global")
        ir.declare_var(y, (n[()],), "int32", "output", "gpu/global")
        "nid: L1"
        for i in range(0, n[()]):
            y[i] = x[i] + 1

    with ir.VarDef("n", (), "int32", "input", "gpu/global") as n:
        with ir.VarDef([
            ("x", (n[()],), "int32", "input", "gpu/global"),
            ("y", (n[()],), "int32", "output", "gpu/global"),
        ]) as (x, y):
            with ir.For("i", 0, n[()], nid="L1") as i:
                y[i] = x[i] + 1
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L1", "threadIdx.x")
    with pytest.raises(ir.InvalidProgram):
        print(ir.lower(s.func(), target))


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
    assert "atomicAdd" in code
    assert "+=" not in code
    print(ir.debug.with_line_no(code))
    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.sum(x_np, axis=1)
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
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.sum(x_np, axis=1)
    assert np.array_equal(y_np, y_std)


def test_unroll_for():

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
    s.unroll("L2")
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
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.sum(x_np, axis=1)
    assert np.array_equal(y_np, y_std)


def test_vectorize():
    with ir.VarDef([
        ("x", (4, 64), "int32", "input", "gpu/global"),
        ("y", (4, 64), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 64, nid="L2") as j:
                y[i, j] = x[i, j] * 2
    func = ir.Func("main", ["x", "y"], ir.pop_ast())

    s = ir.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    assert "int4" in code

    x_np = np.random.randint(0, 100, (4, 64)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(4, 64)

    y_std = x_np * 2
    assert np.array_equal(y_np, y_std)


def test_vectorize_with_non_vector_access():
    with ir.VarDef([
        ("x", (4,), "int32", "input", "gpu/global"),
        ("y", (4, 64), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 64, nid="L2") as j:
                y[i, j] = x[i] * 2
    func = ir.Func("main", ["x", "y"], ir.pop_ast())

    s = ir.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    assert "int4" in code

    x_np = np.random.randint(0, 100, (4,)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(4, 64)

    y_std = np.broadcast_to(x_np * 2, (64, 4)).transpose()
    assert np.array_equal(y_np, y_std)


def test_vectorize_use_iter():
    with ir.VarDef("y", (4, 64), "int32", "output", "gpu/global") as y:
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 64, nid="L2") as j:
                y[i, j] = i + j
    func = ir.Func("main", ["y"], ir.pop_ast())

    s = ir.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    assert "int4" in code

    y_np = np.zeros((4, 64), dtype="int32")
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(y=y_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(4, 64)

    y_std = np.array([[i + j for j in range(64)] for i in range(4)])
    assert np.array_equal(y_np, y_std)


def test_vectorize_fallback_to_shorter_when_not_divisible():
    with ir.VarDef([
        ("x", (4, 62), "int32", "input", "gpu/global"),
        ("y", (4, 62), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 62, nid="L2") as j:
                y[i, j] = x[i, j] * 2
    func = ir.Func("main", ["x", "y"], ir.pop_ast())

    s = ir.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    assert "int2" in code

    x_np = np.random.randint(0, 100, (4, 62)).astype("int32")
    y_np = np.zeros((4, 62), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(4, 62)

    y_std = x_np * 2
    assert np.array_equal(y_np, y_std)


def test_vectorize_fallback_to_shorter_when_not_aligned():
    with ir.VarDef([
        ("x", (4, 66), "int32", "input", "gpu/global"),
        ("y", (4, 64), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 64, nid="L2") as j:
                y[i, j] = x[i, j + 2] * 2
    func = ir.Func("main", ["x", "y"], ir.pop_ast())

    s = ir.Schedule(func)
    s.parallelize("L1", "blockIdx.x")
    s.vectorize("L2")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    assert "int2" in code

    x_np = np.random.randint(0, 100, (4, 66)).astype("int32")
    y_np = np.zeros((4, 64), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(4, 64)

    y_std = x_np[:, 2:] * 2
    assert np.array_equal(y_np, y_std)


def test_cublas_basic():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (48, 64), "float32", "input", "cpu")
        ir.declare_var(b, (64, 72), "float32", "input", "cpu")
        ir.declare_var(c, (48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[i, k] * b[k, j]

    s = ir.Schedule(test)
    s.as_matmul("L1")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    assert "cublas" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ir.Array(a_np, device)
    b_arr = ir.Array(b_np, device)
    c_arr = ir.Array(c_np, device)
    driver = ir.Driver(func, code, device)
    driver.set_params(a=a_arr, b=b_arr, c=c_arr)
    driver.run()
    c_result = c_arr.numpy().reshape(48, 72)

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))
