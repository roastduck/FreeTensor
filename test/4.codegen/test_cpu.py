import ir
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
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_parallel_reduction():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 64), "int32", "input", "cpu")
        ir.declare_var(y, (4,), "int32", "inout", "cpu")
        "nid: L1"
        for i in range(0, 4):
            "nid: L2"
            for j in range(0, 64):
                y[i] = y[i] + x[i, j]

    with ir.VarDef([("x", (4, 64), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "inout", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 64, nid="L2") as j:
                y[i] = y[i] + x[i, j]
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L2", "openmp")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    assert "#pragma omp atomic" in code
    assert "+=" in code
    print(code)
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
        ir.declare_var(x, (4, 64), "int32", "input", "cpu")
        ir.declare_var(y, (4,), "int32", "inout", "cpu")
        "nid: L1"
        for i in range(0, 4):
            "nid: L2"
            for j in range(0, 64):
                y[i] = y[i] + x[i, j]

    with ir.VarDef([("x", (4, 64), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "inout", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 64, nid="L2") as j:
                y[i] = y[i] + x[i, j]
    assert ir.pop_ast().match(test.body)

    s = ir.Schedule(test)
    s.parallelize("L1", "openmp")
    func = ir.lower(s.func(), target)
    print(func)

    code = ir.codegen(func, target)
    assert "#pragma omp atomic" not in code
    assert "+=" in code
    print(code)
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
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
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
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)
