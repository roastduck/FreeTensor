import ir
import pytest
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


def test_omp_for_2():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 4), "int32", "input", "cpu")
        ir.declare_var(y, (4, 4), "int32", "output", "cpu")
        "nid: L1"
        for i in range(4):
            "nid: L2"
            for j in range(4):
                y[i, j] = x[i, j] + 1

    s = ir.Schedule(test)
    L12 = s.merge("L1", "L2")
    s.parallelize(L12, "openmp")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    x_np = np.array([[i + j for j in range(4)] for i in range(4)],
                    dtype="int32")
    y_np = np.zeros((4, 4), dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(target))
    y_arr = ir.Array(y_np, ir.Device(target))
    driver = ir.Driver(func, code, device)
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(4, 4)

    y_std = np.array([[i + j + 1 for j in range(4)] for i in range(4)],
                     dtype="int32")
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


def test_serial_reduction_2():

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4, 64), "int32", "input", "cpu")
        ir.declare_var(y, (4,), "int32", "output", "cpu")
        "nid: L1"
        for i in range(0, 4):
            local_sum = ir.create_var((), "int32", "cache", "cpu")
            local_sum[()] = 0.
            "nid: L2"
            for j in range(0, 64):
                local_sum[()] += x[i, j]
            y[i] = local_sum[()] * 2

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

    y_std = np.sum(x_np, axis=1) * 2
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


@pytest.mark.skipif(not ir.with_mkl(), reason="requires MKL")
def test_mkl_basic():

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
    assert "cblas" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
    b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
    c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(a=a_arr, b=b_arr, c=c_arr)
    driver.run()
    c_result = c_arr.numpy().reshape(48, 72)

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


@pytest.mark.skipif(not ir.with_mkl(), reason="requires MKL")
def test_mkl_trans_a():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (64, 48), "float32", "input", "cpu")
        ir.declare_var(b, (64, 72), "float32", "input", "cpu")
        ir.declare_var(c, (48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[k, i] * b[k, j]

    s = ir.Schedule(test)
    s.as_matmul("L1")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    assert "cblas" in code
    a_np = np.random.uniform(size=(64, 48)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
    b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
    c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(a=a_arr, b=b_arr, c=c_arr)
    driver.run()
    c_result = c_arr.numpy().reshape(48, 72)

    assert np.all(np.isclose(c_result, c_np + a_np.transpose() @ b_np))


@pytest.mark.skipif(not ir.with_mkl(), reason="requires MKL")
def test_mkl_trans_b():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (48, 64), "float32", "input", "cpu")
        ir.declare_var(b, (72, 64), "float32", "input", "cpu")
        ir.declare_var(c, (48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[i, k] * b[j, k]

    s = ir.Schedule(test)
    s.as_matmul("L1")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    assert "cblas" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(72, 64)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
    b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
    c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(a=a_arr, b=b_arr, c=c_arr)
    driver.run()
    c_result = c_arr.numpy().reshape(48, 72)

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np.transpose()))


@pytest.mark.skipif(not ir.with_mkl(), reason="requires MKL")
def test_mkl_trans_c():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (48, 64), "float32", "input", "cpu")
        ir.declare_var(b, (64, 72), "float32", "input", "cpu")
        ir.declare_var(c, (72, 48), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[j, i] += a[i, k] * b[k, j]

    s = ir.Schedule(test)
    s.as_matmul("L1")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    assert "cblas" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(72, 48)).astype("float32")
    a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
    b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
    c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(a=a_arr, b=b_arr, c=c_arr)
    driver.run()
    c_result = c_arr.numpy().reshape(72, 48)

    assert np.all(np.isclose(c_result, c_np + (a_np @ b_np).transpose()))


@pytest.mark.skipif(not ir.with_mkl(), reason="requires MKL")
def test_mkl_batch():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (4, 48, 64), "float32", "input", "cpu")
        ir.declare_var(b, (4, 64, 72), "float32", "input", "cpu")
        ir.declare_var(c, (4, 48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for n in range(4):
            for i in range(48):
                for j in range(72):
                    for k in range(64):
                        c[n, i, j] += a[n, i, k] * b[n, k, j]

    s = ir.Schedule(test)
    s.as_matmul("L1")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    assert "cblas" in code
    a_np = np.random.uniform(size=(4, 48, 64)).astype("float32")
    b_np = np.random.uniform(size=(4, 64, 72)).astype("float32")
    c_np = np.random.uniform(size=(4, 48, 72)).astype("float32")
    a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
    b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
    c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(a=a_arr, b=b_arr, c=c_arr)
    driver.run()
    c_result = c_arr.numpy().reshape(4, 48, 72)

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


@pytest.mark.skipif(not ir.with_mkl(), reason="requires MKL")
def test_mkl_splitted_dim():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (48, 16, 4), "float32", "input", "cpu")
        ir.declare_var(b, (16, 4, 72), "float32", "input", "cpu")
        ir.declare_var(c, (48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for k0 in range(16):
            for i in range(48):
                for j in range(72):
                    for k1 in range(4):
                        c[i, j] += a[i, k0, k1] * b[k0, k1, j]

    s = ir.Schedule(test)
    s.as_matmul("L1")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    assert "cblas" in code
    a_np = np.random.uniform(size=(48, 16, 4)).astype("float32")
    b_np = np.random.uniform(size=(16, 4, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
    b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
    c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(a=a_arr, b=b_arr, c=c_arr)
    driver.run()
    c_result = c_arr.numpy().reshape(48, 72)

    assert np.all(
        np.isclose(c_result,
                   c_np + a_np.reshape(48, 64) @ b_np.reshape(64, 72)))


@pytest.mark.skipif(not ir.with_mkl(), reason="requires MKL")
def test_mkl_with_init():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (48, 64), "float32", "input", "cpu")
        ir.declare_var(b, (64, 72), "float32", "input", "cpu")
        ir.declare_var(c, (48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for j in range(72):
                c[i, j] = 0
                for k in range(64):
                    c[i, j] += a[i, k] * b[k, j]

    s = ir.Schedule(test)
    s.as_matmul("L1")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    assert "cblas" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
    b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
    c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(a=a_arr, b=b_arr, c=c_arr)
    driver.run()
    c_result = c_arr.numpy().reshape(48, 72)

    assert np.all(np.isclose(c_result, a_np @ b_np))
