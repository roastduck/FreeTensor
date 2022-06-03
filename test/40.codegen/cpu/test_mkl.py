import freetensor as ft
import pytest
import numpy as np

if not ft.with_mkl():
    pytest.skip("requires MKL", allow_module_level=True)

target = ft.CPU()
device = ft.Device(target)


def test_mkl_basic():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 64), "float32", "input", "cpu"]
        b: ft.Var[(64, 72), "float32", "input", "cpu"]
        c: ft.Var[(48, 72), "float32", "inout", "cpu"]
        #! nid: L1
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[i, k] * b[k, j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cblas" in str(code)
    assert "mkl_set_num_threads_local(0)" in str(code)
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


def test_mkl_reverse_idx():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 64), "float32", "input", "cpu"]
        b: ft.Var[(64, 72), "float32", "input", "cpu"]
        c: ft.Var[(48, 72), "float32", "inout", "cpu"]
        #! nid: L1
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[47 - i, 71 - j] += a[47 - i, 63 - k] * b[63 - k, 71 - j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cblas" in str(code)
    assert "mkl_set_num_threads_local(0)" in str(code)
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


def test_mkl_trans_a():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(64, 48), "float32", "input", "cpu"]
        b: ft.Var[(64, 72), "float32", "input", "cpu"]
        c: ft.Var[(48, 72), "float32", "inout", "cpu"]
        #! nid: L1
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[k, i] * b[k, j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cblas" in str(code)
    a_np = np.random.uniform(size=(64, 48)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np.transpose() @ b_np))


def test_mkl_trans_b():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 64), "float32", "input", "cpu"]
        b: ft.Var[(72, 64), "float32", "input", "cpu"]
        c: ft.Var[(48, 72), "float32", "inout", "cpu"]
        #! nid: L1
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[i, k] * b[j, k]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cblas" in str(code)
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(72, 64)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np.transpose()))


def test_mkl_trans_c():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 64), "float32", "input", "cpu"]
        b: ft.Var[(64, 72), "float32", "input", "cpu"]
        c: ft.Var[(72, 48), "float32", "inout", "cpu"]
        #! nid: L1
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[j, i] += a[i, k] * b[k, j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cblas" in str(code)
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(72, 48)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + (a_np @ b_np).transpose()))


def test_mkl_batch():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(4, 48, 64), "float32", "input", "cpu"]
        b: ft.Var[(4, 64, 72), "float32", "input", "cpu"]
        c: ft.Var[(4, 48, 72), "float32", "inout", "cpu"]
        #! nid: L1
        for n in range(4):
            for i in range(48):
                for j in range(72):
                    for k in range(64):
                        c[n, i, j] += a[n, i, k] * b[n, k, j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cblas" in str(code)
    a_np = np.random.uniform(size=(4, 48, 64)).astype("float32")
    b_np = np.random.uniform(size=(4, 64, 72)).astype("float32")
    c_np = np.random.uniform(size=(4, 48, 72)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


def test_mkl_splitted_dim():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 16, 4), "float32", "input", "cpu"]
        b: ft.Var[(16, 4, 72), "float32", "input", "cpu"]
        c: ft.Var[(48, 72), "float32", "inout", "cpu"]
        #! nid: L1
        for k0 in range(16):
            for i in range(48):
                for j in range(72):
                    for k1 in range(4):
                        c[i, j] += a[i, k0, k1] * b[k0, k1, j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cblas" in str(code)
    a_np = np.random.uniform(size=(48, 16, 4)).astype("float32")
    b_np = np.random.uniform(size=(16, 4, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(
        np.isclose(c_result,
                   c_np + a_np.reshape(48, 64) @ b_np.reshape(64, 72)))


def test_mkl_with_init():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 64), "float32", "input", "cpu"]
        b: ft.Var[(64, 72), "float32", "input", "cpu"]
        c: ft.Var[(48, 72), "float32", "inout", "cpu"]
        #! nid: L1
        for i in range(48):
            for j in range(72):
                c[i, j] = 0
                for k in range(64):
                    c[i, j] += a[i, k] * b[k, j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cblas" in str(code)
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, a_np @ b_np))


def test_mkl_in_parallel():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(64, 48, 64), "float32", "input", "cpu"]
        b: ft.Var[(64, 64, 72), "float32", "input", "cpu"]
        c: ft.Var[(64, 48, 72), "float32", "inout", "cpu"]
        #! nid: L1
        for n in range(64):
            #! nid: L2
            for i in range(48):
                for j in range(72):
                    for k in range(64):
                        c[n, i, j] += a[n, i, k] * b[n, k, j]

    s = ft.Schedule(test)
    s.as_matmul("L2")
    s.parallelize("L1", "openmp")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cblas" in str(code)
    assert "mkl_set_num_threads_local(1)" in str(code)
    a_np = np.random.uniform(size=(64, 48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 64, 72)).astype("float32")
    c_np = np.random.uniform(size=(64, 48, 72)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


def test_mkl_matrix_vector():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 64), "float32", "input", "cpu"]
        b: ft.Var[(64,), "float32", "input", "cpu"]
        c: ft.Var[(48,), "float32", "inout", "cpu"]
        #! nid: L1
        for i in range(48):
            for k in range(64):
                c[i] += a[i, k] * b[k]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cblas" in str(code)
    assert "mkl_set_num_threads_local(0)" in str(code)
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64,)).astype("float32")
    c_np = np.random.uniform(size=(48,)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


def test_mkl_vector_matrix():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(64,), "float32", "input", "cpu"]
        b: ft.Var[(64, 48), "float32", "input", "cpu"]
        c: ft.Var[(48,), "float32", "inout", "cpu"]
        #! nid: L1
        for i in range(48):
            for k in range(64):
                c[i] += a[k] * b[k, i]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cblas" in str(code)
    assert "mkl_set_num_threads_local(0)" in str(code)
    a_np = np.random.uniform(size=(64,)).astype("float32")
    b_np = np.random.uniform(size=(64, 48)).astype("float32")
    c_np = np.random.uniform(size=(48,)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


def test_mkl_vardef_in_loop():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 64), "float32", "input", "cpu"]
        c: ft.Var[(48, 72), "float32", "inout", "cpu"]
        #! nid: L1
        for i in range(48):
            b: ft.Var[(64, 72), "float32", "input", "cpu"]
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[i, k] * b[k, j]

    s = ft.Schedule(test)
    s.as_matmul("L1")
    print(s.ast())
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cblas" in str(code)
    assert "mkl_set_num_threads_local(0)" in str(code)
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))
