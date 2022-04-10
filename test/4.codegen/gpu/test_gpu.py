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
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_define_output_inside_kernel():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4,), "int32", "input", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            ir.declare_var(y, (4,), "int32", "output", "gpu/global")
            y[i] = x[i] + 1

    with ir.VarDef("x", (4,), "int32", "input", "gpu/global") as x:
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.VarDef("y", (4,), "int32", "output", "gpu/global") as y:
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
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
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
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
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
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_global_mem():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4,), "int32", "input", "gpu/global")
        ir.declare_var(y, (4,), "int32", "output", "gpu/global")
        t = ir.create_var((4,), "int32", "gpu/global")
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
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([3, 5, 7, 9], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_global_mem_in_kernel():

    @ir.transform
    def test(x, y1, y2):
        ir.declare_var(x, (4,), "int32", "input", "gpu/global")
        ir.declare_var(y1, (4,), "int32", "output", "gpu/global")
        ir.declare_var(y2, (4,), "int32", "output", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            t = ir.create_var((), "int32", "gpu/global")
            t[()] = x[i] * 2
            y1[i] = t[()] + 1
            y2[i] = t[()] + 2

    s = ir.Schedule(test)
    s.parallelize("L1", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    assert "cudaMalloc" in code
    assert "cudaFree" in code
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y1_np = np.zeros((4,), dtype="int32")
    y2_np = np.zeros((4,), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y1_arr = ir.Array(y1_np, device)
    y2_arr = ir.Array(y2_np, device)
    ir.Driver(func, code, device)(x=x_arr, y1=y1_arr, y2=y2_arr)
    y1_np = y1_arr.numpy()
    y2_np = y2_arr.numpy()

    y1_std = np.array([3, 5, 7, 9], dtype="int32")
    y2_std = np.array([4, 6, 8, 10], dtype="int32")
    assert np.array_equal(y1_np, y1_std)
    assert np.array_equal(y2_np, y2_std)


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
    ir.Driver(func, code, device)(n=n_arr, x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

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
    ir.Driver(func, code, device)(n=n_arr, x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

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

    s = ir.Schedule(ir.Func("main", ["n", "x", "y"], [], ir.pop_ast()))
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
    ir.Driver(func, code, device)(n=n_arr, x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = x_np + 1
    assert np.array_equal(y_np, y_std)


def test_use_cpu_iters():
    with ir.VarDef("y", (4, 1000), "int32", "output", "gpu/global") as y:
        with ir.For("i", 0, 4, nid="Li") as i:
            with ir.For("j", 0, 1000, nid="Lj") as j:
                y[i, j] = i + j

    s = ir.Schedule(ir.Func("main", ["y"], [], ir.pop_ast()))
    s.parallelize('Lj', "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    y_np = np.zeros((4, 1000), dtype="int32")
    y_arr = ir.Array(y_np, device)
    ir.Driver(func, code, device)(y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([[i + j for j in range(1000)] for i in range(4)],
                     dtype="int32")
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
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array(np.sin(x_np), dtype="float32")
    assert np.all(np.isclose(y_np, y_std))


def test_multiplex_shared_1():

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
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([range(1, 513, 2)] * 4, dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_multiplex_shared_2():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 256), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((256,), "int32", "gpu/shared")
            "nid: L1"
            for j in range(i * 64, (i + 1) * 64):
                t[j] = x[i, j] * 2
                # No need to hoist over i, although i is not present here
            "nid: L2"
            for j in range(i * 64, (i + 1) * 64):
                y[i, j] = t[j] + 1

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
            with ir.For(".threadIdx.x", 0, 64) as j:
                with ir.VarDef("t", (256,), "int32", "cache",
                               "gpu/shared") as t:
                    t[j + i * 64] = x[i, j + i * 64] * 2
                    y[i, j + i * 64] = t[j + i * 64] + 1
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_simplex_local_1():

    @ir.transform
    def test(x, y, z):
        ir.declare_var(x, (10, 10, 10), "int32", "input", "gpu/global")
        ir.declare_var(y, (10, 10, 10), "int32", "output", "gpu/global")
        ir.declare_var(z, (10, 10, 10), "int32", "output", "gpu/global")
        'nid: Lb'
        for b in range(10):
            'nid: t'
            t = ir.create_var((10, 10), "int32", "gpu/global")
            'nid: L0'
            for i in range(10):
                for j in range(10):
                    t[i, j] = x[b, i, j] * 2
            'nid: L1'
            for i in range(10):
                for j in range(10):
                    y[b, i, j] = t[i, j] + 1
            'nid: L2'
            for i in range(10):
                for j in range(10):
                    z[b, i, j] = t[i, j] + 2

    s = ir.Schedule(test)
    s.parallelize("Lb", "blockIdx.x")
    s.parallelize("L0", "threadIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.parallelize("L2", "threadIdx.x")
    s.set_mem_type("t", "gpu/local")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([("x", (10, 10, 10), "int32", "input", "gpu/global"),
                    ("y", (10, 10, 10), "int32", "output", "gpu/global"),
                    ("z", (10, 10, 10), "int32", "output", "gpu/global")
                   ]) as (x, y, z):
        with ir.For(".blockIdx", 0, 10) as b:
            with ir.For(".threadIdx.x", 0, 10) as i:
                with ir.VarDef("t", (10,), "int32", "cache", "gpu/local") as t:
                    with ir.For("j", 0, 10) as j:
                        t[j] = x[b, i, j] * 2
                    with ir.For("j$1", 0, 10) as j:
                        y[b, i, j] = t[j] + 1
                    with ir.For("j$2", 0, 10) as j:
                        z[b, i, j] = t[j] + 2
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)

    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    x_np = np.random.randint(0, 100, (10, 10, 10)).astype("int32")
    y_np = np.zeros((10, 10, 10), dtype="int32")
    z_np = np.zeros((10, 10, 10), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    z_arr = ir.Array(z_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr, z=z_arr)
    y_np = y_arr.numpy()
    z_np = z_arr.numpy()

    assert np.array_equal(y_np, x_np * 2 + 1)
    assert np.array_equal(z_np, x_np * 2 + 2)


def test_simplex_local_2():

    @ir.transform
    def test(x, y, z):
        ir.declare_var(x, (10, 10, 10), "int32", "input", "gpu/global")
        ir.declare_var(y, (10, 10, 10), "int32", "output", "gpu/global")
        'nid: Lb'
        for b in range(10):
            'nid: t'
            t = ir.create_var((10, 10), "int32", "gpu/global")
            'nid: L0'
            for i in range(10):
                for j in range(10):
                    t[i, j] = x[b, i, j] * 2
                for j in range(10):
                    t[i, j] += t[i, i]
                    # The last dimension can be removed although accessed with i
            'nid: L1'
            for i in range(10):
                for j in range(10):
                    y[b, i, j] = t[i, j] + 1

    s = ir.Schedule(test)
    s.parallelize("Lb", "blockIdx.x")
    s.parallelize("L0", "threadIdx.x")
    s.parallelize("L1", "threadIdx.x")
    s.set_mem_type("t", "gpu/local")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("x", (10, 10, 10), "int32", "input", "gpu/global"),
        ("y", (10, 10, 10), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ir.For(".blockIdx", 0, 10) as b:
            with ir.For(".threadIdx.x", 0, 10) as i:
                with ir.VarDef("t", (10,), "int32", "cache", "gpu/local") as t:
                    with ir.For("j", 0, 10) as j:
                        t[j] = x[b, i, j] * 2
                    with ir.For("j$1", 0, 10) as j:
                        t[j] += t[i]
                    with ir.For("j$2", 0, 10) as j:
                        y[b, i, j] = t[j] + 1
    assert ir.make_1d_var(ir.make_reduction(ir.pop_ast())).match(func.body)


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

    s = ir.Schedule(ir.Func("main", ["n", "x", "y"], [], ir.pop_ast()))
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
    ir.Driver(func, code, device)(n=n_arr, x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

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
            t = ir.create_var((4,), "int32", "gpu/shared")
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
                    ir.Eval(ir.intrinsic("__syncwarp()", has_side_effect=True))
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
    ir.Driver(func, code, device)(a=a_arr, b=b_arr, c=c_arr)
    c_np = c_arr.numpy()

    c_std = a_np @ b_np
    assert np.array_equal(c_np, c_std)


def test_bounded_length():

    @ir.transform
    def test(a, b):
        ir.declare_var(a, (100, 100), "int32", "input", "gpu/global")
        ir.declare_var(b, (100, 100), "int32", "output", "gpu/global")
        'nid: Li'
        for i in range(100):
            'nid: Lj'
            for j in range(i):
                b[i, j] = a[i, j] + 1

    s = ir.Schedule(test)
    s.parallelize("Lj", "threadIdx.y")
    s.parallelize("Li", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    with ir.VarDef([
        ("a", (100, 100), "int32", "input", "gpu/global"),
        ("b", (100, 100), "int32", "output", "gpu/global"),
    ]) as (a, b):
        with ir.For(".threadIdx.y", 0, 99) as thy:
            with ir.For(".threadIdx.x", 1, 100) as thx:
                with ir.If(thx >= thy + 1):
                    b[thx, thy] = a[thx, thy] + 1
    assert ir.make_1d_var(ir.pop_ast()).match(func.body)


def test_parallel_broadcast():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (4, 1), "int32", "input", "gpu/global")
        ir.declare_var(b, (1, 8), "int32", "input", "gpu/global")
        ir.declare_var(c, (4, 8), "int32", "output", "gpu/global")
        "nid: L0"
        for i in range(0, 4):
            t = ir.create_var((1,), "int32", "gpu/shared")
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
                    ir.Eval(ir.intrinsic("__syncwarp()", has_side_effect=True))
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
    ir.Driver(func, code, device)(a=a_arr, b=b_arr, c=c_arr)
    c_np = c_arr.numpy()

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
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.sum(x_np, axis=1)
    assert np.array_equal(y_np, y_std)


def test_streams():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 256), "int32", "input", "gpu/global")
        ir.declare_var(y, (4, 256), "int32", "output", "gpu/global")
        "nid: L1"
        for i in range(0, 4):
            "nid: L2"
            for j in range(0, 256):
                y[i, j] = x[i, j] + 1
            "nid: L3"
            for j in range(0, 128):
                y[i, j * 2] *= 2

    s = ir.Schedule(test)
    s.parallelize("L1", "cudaStream")
    s.parallelize("L2", "threadIdx.x")
    s.parallelize("L3", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(ir.debug.with_line_no(code))
    assert "cudaStreamCreate" in code
    x_np = np.random.randint(0, 100, (4, 256)).astype("int32")
    y_np = np.zeros((4, 256), dtype="int32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    ir.Driver(func, code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = ((x_np + 1).reshape(4, 128, 2) * [2, 1]).reshape(4, 256)
    assert np.array_equal(y_np, y_std)


def test_merge_no_deps_1():

    @ir.transform
    def test(ptr, edge1, edge2):
        ir.declare_var(ptr, (4, 11), "int32", "input", "cpu")
        ir.declare_var(edge1, (4, 50), "int32", "input", "cpu")
        ir.declare_var(edge2, (4, 50), "int32", "output", "cpu")
        'nid: Lb'
        for b in range(4):
            'nid: Li1'
            'no_deps: edge2'
            for i in range(10):
                for j in range(ptr[b, i], ptr[b, i + 1]):
                    edge2[b, j] = edge1[b, j] + i
            'nid: Li2'
            'no_deps: edge2'
            for i in range(10):
                for j in range(ptr[b, i], ptr[b, i + 1]):
                    edge2[b, j] += j

    print(test)
    s = ir.Schedule(test)
    s.parallelize("Lb", "blockIdx.x")
    s.parallelize("Li1", "threadIdx.x")
    s.parallelize("Li2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    def matcher(x):
        if x.type() == ir.ASTNodeType.For:
            node = x
            if str(node.property.parallel) == "threadIdx.x":
                return True
        return False

    checker = ir.Schedule(func)
    assert checker.find(matcher).property.no_deps == ['edge2']


def test_merge_no_deps_2():

    @ir.transform
    def test(ptr, edge1, edge2):
        ir.declare_var(ptr, (4, 11), "int32", "input", "cpu")
        ir.declare_var(edge1, (4, 50), "int32", "input", "cpu")
        ir.declare_var(edge2, (4, 50), "int32", "output", "cpu")
        ir.declare_var(foobar, (
            4,
            10,
        ), "int32", "output", "cpu")
        'nid: Lb'
        for b in range(4):
            'nid: Li1'
            'no_deps: edge2'
            for i in range(10):
                for j in range(ptr[b, i], ptr[b, i + 1]):
                    edge2[b, j] = edge1[b, j] + i
            'nid: Li2'
            for i in range(10):
                # Nothing to do with edge2 here
                foobar[b, i] = i

    print(test)
    s = ir.Schedule(test)
    s.parallelize("Lb", "blockIdx.x")
    s.parallelize("Li1", "threadIdx.x")
    s.parallelize("Li2", "threadIdx.x")
    func = ir.lower(s.func(), target)
    print(func)

    def matcher(x):
        if x.type() == ir.ASTNodeType.For:
            node = x
            if str(node.property.parallel) == "threadIdx.x":
                return True
        return False

    checker = ir.Schedule(func)
    assert checker.find(matcher).property.no_deps == ['edge2']


def test_merge_no_deps_3():

    @ir.transform
    def test(ptr, edge1, edge2):
        ir.declare_var(ptr, (4, 11), "int32", "input", "cpu")
        ir.declare_var(edge1, (4, 50), "int32", "input", "cpu")
        ir.declare_var(edge2, (4, 50), "int32", "output", "cpu")
        'nid: Lb'
        for b in range(4):
            'nid: Li1'
            'no_deps: edge2'
            for i in range(10):
                for j in range(ptr[b, i], ptr[b, i + 1]):
                    edge2[b, j] = edge1[b, j] + i
            'nid: Li2'  # If we don't mark edge2 here
            for i in range(10):
                for j in range(ptr[b, i], ptr[b, i + 1] + 1):
                    edge2[b, j] = edge2[b, j] * 2 + j

    print(test)
    s = ir.Schedule(test)
    s.parallelize("Lb", "blockIdx.x")
    s.parallelize("Li1", "threadIdx.x")
    with pytest.raises(ir.InvalidSchedule):
        s.parallelize("Li2", "threadIdx.x")
