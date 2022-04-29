import freetensor as ft
import pytest
import numpy as np

target = ft.CPU()
device = ft.Device(target)


@pytest.mark.skipif(not ft.with_mkl(), reason="requires MKL")
def test_mkl_basic():

    @ft.transform
    def test(a, b, c):
        ft.declare_var(a, (48, 64), "float32", "input", "cpu")
        ft.declare_var(b, (64, 72), "float32", "input", "cpu")
        ft.declare_var(c, (48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[i, k] * b[k, j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "cblas" in code
    assert "mkl_set_num_threads_local(0)" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np, ft.Device(ft.CPU()))
    b_arr = ft.Array(b_np, ft.Device(ft.CPU()))
    c_arr = ft.Array(c_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


@pytest.mark.skipif(not ft.with_mkl(), reason="requires MKL")
def test_mkl_reverse_idx():

    @ft.transform
    def test(a, b, c):
        ft.declare_var(a, (48, 64), "float32", "input", "cpu")
        ft.declare_var(b, (64, 72), "float32", "input", "cpu")
        ft.declare_var(c, (48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[47 - i, 71 - j] += a[47 - i, 63 - k] * b[63 - k, 71 - j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "cblas" in code
    assert "mkl_set_num_threads_local(0)" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np, ft.Device(ft.CPU()))
    b_arr = ft.Array(b_np, ft.Device(ft.CPU()))
    c_arr = ft.Array(c_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


@pytest.mark.skipif(not ft.with_mkl(), reason="requires MKL")
def test_mkl_trans_a():

    @ft.transform
    def test(a, b, c):
        ft.declare_var(a, (64, 48), "float32", "input", "cpu")
        ft.declare_var(b, (64, 72), "float32", "input", "cpu")
        ft.declare_var(c, (48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[k, i] * b[k, j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "cblas" in code
    a_np = np.random.uniform(size=(64, 48)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np, ft.Device(ft.CPU()))
    b_arr = ft.Array(b_np, ft.Device(ft.CPU()))
    c_arr = ft.Array(c_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np.transpose() @ b_np))


@pytest.mark.skipif(not ft.with_mkl(), reason="requires MKL")
def test_mkl_trans_b():

    @ft.transform
    def test(a, b, c):
        ft.declare_var(a, (48, 64), "float32", "input", "cpu")
        ft.declare_var(b, (72, 64), "float32", "input", "cpu")
        ft.declare_var(c, (48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[i, k] * b[j, k]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "cblas" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(72, 64)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np, ft.Device(ft.CPU()))
    b_arr = ft.Array(b_np, ft.Device(ft.CPU()))
    c_arr = ft.Array(c_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np.transpose()))


@pytest.mark.skipif(not ft.with_mkl(), reason="requires MKL")
def test_mkl_trans_c():

    @ft.transform
    def test(a, b, c):
        ft.declare_var(a, (48, 64), "float32", "input", "cpu")
        ft.declare_var(b, (64, 72), "float32", "input", "cpu")
        ft.declare_var(c, (72, 48), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[j, i] += a[i, k] * b[k, j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "cblas" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(72, 48)).astype("float32")
    a_arr = ft.Array(a_np, ft.Device(ft.CPU()))
    b_arr = ft.Array(b_np, ft.Device(ft.CPU()))
    c_arr = ft.Array(c_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + (a_np @ b_np).transpose()))


@pytest.mark.skipif(not ft.with_mkl(), reason="requires MKL")
def test_mkl_batch():

    @ft.transform
    def test(a, b, c):
        ft.declare_var(a, (4, 48, 64), "float32", "input", "cpu")
        ft.declare_var(b, (4, 64, 72), "float32", "input", "cpu")
        ft.declare_var(c, (4, 48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for n in range(4):
            for i in range(48):
                for j in range(72):
                    for k in range(64):
                        c[n, i, j] += a[n, i, k] * b[n, k, j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "cblas" in code
    a_np = np.random.uniform(size=(4, 48, 64)).astype("float32")
    b_np = np.random.uniform(size=(4, 64, 72)).astype("float32")
    c_np = np.random.uniform(size=(4, 48, 72)).astype("float32")
    a_arr = ft.Array(a_np, ft.Device(ft.CPU()))
    b_arr = ft.Array(b_np, ft.Device(ft.CPU()))
    c_arr = ft.Array(c_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


@pytest.mark.skipif(not ft.with_mkl(), reason="requires MKL")
def test_mkl_splitted_dim():

    @ft.transform
    def test(a, b, c):
        ft.declare_var(a, (48, 16, 4), "float32", "input", "cpu")
        ft.declare_var(b, (16, 4, 72), "float32", "input", "cpu")
        ft.declare_var(c, (48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for k0 in range(16):
            for i in range(48):
                for j in range(72):
                    for k1 in range(4):
                        c[i, j] += a[i, k0, k1] * b[k0, k1, j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "cblas" in code
    a_np = np.random.uniform(size=(48, 16, 4)).astype("float32")
    b_np = np.random.uniform(size=(16, 4, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np, ft.Device(ft.CPU()))
    b_arr = ft.Array(b_np, ft.Device(ft.CPU()))
    c_arr = ft.Array(c_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(
        np.isclose(c_result,
                   c_np + a_np.reshape(48, 64) @ b_np.reshape(64, 72)))


@pytest.mark.skipif(not ft.with_mkl(), reason="requires MKL")
def test_mkl_with_init():

    @ft.transform
    def test(a, b, c):
        ft.declare_var(a, (48, 64), "float32", "input", "cpu")
        ft.declare_var(b, (64, 72), "float32", "input", "cpu")
        ft.declare_var(c, (48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for j in range(72):
                c[i, j] = 0
                for k in range(64):
                    c[i, j] += a[i, k] * b[k, j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "cblas" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np, ft.Device(ft.CPU()))
    b_arr = ft.Array(b_np, ft.Device(ft.CPU()))
    c_arr = ft.Array(c_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, a_np @ b_np))


@pytest.mark.skipif(not ft.with_mkl(), reason="requires MKL")
def test_mkl_in_parallel():

    @ft.transform
    def test(a, b, c):
        ft.declare_var(a, (64, 48, 64), "float32", "input", "cpu")
        ft.declare_var(b, (64, 64, 72), "float32", "input", "cpu")
        ft.declare_var(c, (64, 48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for n in range(64):
            "nid: L2"
            for i in range(48):
                for j in range(72):
                    for k in range(64):
                        c[n, i, j] += a[n, i, k] * b[n, k, j]

    s = ft.Schedule(test)
    s.as_matmul("L2")
    s.parallelize("L1", "openmp")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "cblas" in code
    assert "mkl_set_num_threads_local(1)" in code
    a_np = np.random.uniform(size=(64, 48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 64, 72)).astype("float32")
    c_np = np.random.uniform(size=(64, 48, 72)).astype("float32")
    a_arr = ft.Array(a_np, ft.Device(ft.CPU()))
    b_arr = ft.Array(b_np, ft.Device(ft.CPU()))
    c_arr = ft.Array(c_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


@pytest.mark.skipif(not ft.with_mkl(), reason="requires MKL")
def test_mkl_matrix_vector():

    @ft.transform
    def test(a, b, c):
        ft.declare_var(a, (48, 64), "float32", "input", "cpu")
        ft.declare_var(b, (64,), "float32", "input", "cpu")
        ft.declare_var(c, (48,), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for k in range(64):
                c[i] += a[i, k] * b[k]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "cblas" in code
    assert "mkl_set_num_threads_local(0)" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64,)).astype("float32")
    c_np = np.random.uniform(size=(48,)).astype("float32")
    a_arr = ft.Array(a_np, ft.Device(ft.CPU()))
    b_arr = ft.Array(b_np, ft.Device(ft.CPU()))
    c_arr = ft.Array(c_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


@pytest.mark.skipif(not ft.with_mkl(), reason="requires MKL")
def test_mkl_vector_matrix():

    @ft.transform
    def test(a, b, c):
        ft.declare_var(a, (64,), "float32", "input", "cpu")
        ft.declare_var(b, (64, 48), "float32", "input", "cpu")
        ft.declare_var(c, (48,), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            for k in range(64):
                c[i] += a[k] * b[k, i]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "cblas" in code
    assert "mkl_set_num_threads_local(0)" in code
    a_np = np.random.uniform(size=(64,)).astype("float32")
    b_np = np.random.uniform(size=(64, 48)).astype("float32")
    c_np = np.random.uniform(size=(48,)).astype("float32")
    a_arr = ft.Array(a_np, ft.Device(ft.CPU()))
    b_arr = ft.Array(b_np, ft.Device(ft.CPU()))
    c_arr = ft.Array(c_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


@pytest.mark.skipif(not ft.with_mkl(), reason="requires MKL")
def test_mkl_vardef_in_loop():

    @ft.transform
    def test(a, b, c):
        ft.declare_var(a, (48, 64), "float32", "input", "cpu")
        ft.declare_var(c, (48, 72), "float32", "inout", "cpu")
        "nid: L1"
        for i in range(48):
            ft.declare_var(b, (64, 72), "float32", "input", "cpu")
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[i, k] * b[k, j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    print(s.ast())
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)
    assert "cblas" in code
    assert "mkl_set_num_threads_local(0)" in code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np, ft.Device(ft.CPU()))
    b_arr = ft.Array(b_np, ft.Device(ft.CPU()))
    c_arr = ft.Array(c_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))
