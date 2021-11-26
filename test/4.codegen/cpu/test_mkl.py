import ir
import pytest
import numpy as np

target = ir.CPU()
device = ir.Device(target)


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
    assert "mkl_set_num_threads_local(0)" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
    b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
    c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy().reshape(48, 72)

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


@pytest.mark.skipif(not ir.with_mkl(), reason="requires MKL")
def test_mkl_reverse_idx():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (48, 64), "float32", "input", "cpu")
        ir.declare_var(b, (64, 72), "float32", "input", "cpu")
        ir.declare_var(c, (48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[47 - i, 71 - j] += a[47 - i, 63 - k] * b[63 - k, 71 - j]

    s = ir.Schedule(test)
    s.as_matmul("L1")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    assert "cblas" in code
    assert "mkl_set_num_threads_local(0)" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
    b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
    c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(a=a_arr, b=b_arr, c=c_arr)
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
    ir.Driver(func, code, ir.Device(ir.CPU()))(a=a_arr, b=b_arr, c=c_arr)
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
    ir.Driver(func, code, ir.Device(ir.CPU()))(a=a_arr, b=b_arr, c=c_arr)
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
    ir.Driver(func, code, ir.Device(ir.CPU()))(a=a_arr, b=b_arr, c=c_arr)
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
    ir.Driver(func, code, ir.Device(ir.CPU()))(a=a_arr, b=b_arr, c=c_arr)
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
    ir.Driver(func, code, ir.Device(ir.CPU()))(a=a_arr, b=b_arr, c=c_arr)
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
    ir.Driver(func, code, ir.Device(ir.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy().reshape(48, 72)

    assert np.all(np.isclose(c_result, a_np @ b_np))


@pytest.mark.skipif(not ir.with_mkl(), reason="requires MKL")
def test_mkl_in_parallel():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (64, 48, 64), "float32", "input", "cpu")
        ir.declare_var(b, (64, 64, 72), "float32", "input", "cpu")
        ir.declare_var(c, (64, 48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for n in range(64):
            "nid: L2"
            for i in range(48):
                for j in range(72):
                    for k in range(64):
                        c[n, i, j] += a[n, i, k] * b[n, k, j]

    s = ir.Schedule(test)
    s.as_matmul("L2")
    s.parallelize("L1", "openmp")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    assert "cblas" in code
    assert "mkl_set_num_threads_local(1)" in code
    a_np = np.random.uniform(size=(64, 48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 64, 72)).astype("float32")
    c_np = np.random.uniform(size=(64, 48, 72)).astype("float32")
    a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
    b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
    c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy().reshape(64, 48, 72)

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


@pytest.mark.skipif(not ir.with_mkl(), reason="requires MKL")
def test_mkl_matrix_vector():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (48, 64), "float32", "input", "cpu")
        ir.declare_var(b, (64,), "float32", "input", "cpu")
        ir.declare_var(c, (48,), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for k in range(64):
                c[i] += a[i, k] * b[k]

    s = ir.Schedule(test)
    s.as_matmul("L1")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    assert "cblas" in code
    assert "mkl_set_num_threads_local(0)" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64,)).astype("float32")
    c_np = np.random.uniform(size=(48,)).astype("float32")
    a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
    b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
    c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


@pytest.mark.skipif(not ir.with_mkl(), reason="requires MKL")
def test_mkl_vector_matrix():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (64,), "float32", "input", "cpu")
        ir.declare_var(b, (64, 48), "float32", "input", "cpu")
        ir.declare_var(c, (48,), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for k in range(64):
                c[i] += a[k] * b[k, i]

    s = ir.Schedule(test)
    s.as_matmul("L1")
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    assert "cblas" in code
    assert "mkl_set_num_threads_local(0)" in code
    a_np = np.random.uniform(size=(64,)).astype("float32")
    b_np = np.random.uniform(size=(64, 48)).astype("float32")
    c_np = np.random.uniform(size=(48,)).astype("float32")
    a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
    b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
    c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


@pytest.mark.skipif(not ir.with_mkl(), reason="requires MKL")
def test_mkl_vardef_in_loop():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (48, 64), "float32", "input", "cpu")
        ir.declare_var(c, (48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            ir.declare_var(b, (64, 72), "float32", "input", "cpu")
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[i, k] * b[k, j]

    s = ir.Schedule(test)
    s.as_matmul("L1")
    print(s.ast())
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
    assert "cblas" in code
    assert "mkl_set_num_threads_local(0)" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
    b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
    c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy().reshape(48, 72)

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))
