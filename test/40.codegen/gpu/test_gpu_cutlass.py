import freetensor as ft
from freetensor import debug
import pytest
import numpy as np

if not ft.with_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

device = ft.GPU()
target = device.target()


def test_float64():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 64), "float64", "input", "gpu/global"]
        b: ft.Var[(64, 72), "float64", "input", "gpu/global"]
        c: ft.Var[(48, 72), "float64", "inout", "gpu/global"]
        #! label: L1
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[i, k] * b[k, j]

    s = ft.Schedule(test)
    s.as_matmul("L1", ft.AsMatMulMode.KeepMemLayout, target, "cutlass")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cutlass" in code.code
    a_np = np.random.uniform(size=(48, 64)).astype("float64")
    b_np = np.random.uniform(size=(64, 72)).astype("float64")
    c_np = np.random.uniform(size=(48, 72)).astype("float64")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.build_binary(code, device)(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


def test_float32():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 64), "float32", "input", "gpu/global"]
        b: ft.Var[(64, 72), "float32", "input", "gpu/global"]
        c: ft.Var[(48, 72), "float32", "inout", "gpu/global"]
        #! label: L1
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[i, k] * b[k, j]

    s = ft.Schedule(test)
    s.as_matmul("L1", ft.AsMatMulMode.KeepMemLayout, target, "cutlass")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cutlass" in code.code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.build_binary(code, device)(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))


def test_float16():
    # Not testing float16 I/O here

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 64), "float32", "input", "gpu/global"]
        b: ft.Var[(64, 72), "float32", "input", "gpu/global"]
        c: ft.Var[(48, 72), "float32", "inout", "gpu/global"]
        a16 = ft.empty((48, 64), "float16", "gpu/global")
        b16 = ft.empty((64, 72), "float16", "gpu/global")
        c16 = ft.empty((48, 72), "float16", "gpu/global")
        #! label: La_in
        for i in range(48):
            for j in range(64):
                a16[i, j] = ft.cast(a[i, j], "float16")
        #! label: Lb_in
        for i in range(64):
            for j in range(72):
                b16[i, j] = ft.cast(b[i, j], "float16")
        #! label: Lc_in
        for i in range(48):
            for j in range(72):
                c16[i, j] = ft.cast(c[i, j], "float16")
        #! label: L1
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c16[i, j] += a16[i, k] * b16[k, j]
        #! label: Lc_out
        for i in range(48):
            for j in range(72):
                c[i, j] = ft.cast(c16[i, j], "float32")

    s = ft.Schedule(test)
    s.parallelize("La_in", "blockIdx.x")
    s.parallelize("Lb_in", "blockIdx.x")
    s.parallelize("Lc_in", "blockIdx.x")
    s.parallelize("Lc_out", "blockIdx.x")
    s.as_matmul("L1", ft.AsMatMulMode.KeepMemLayout, target, "cutlass")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cutlass" in code.code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.build_binary(code, device)(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    c_std = (c_np.astype("float16") +
             a_np.astype("float16") @ b_np.astype("float16")).astype("float32")
    assert np.all(np.isclose(c_result, c_std, atol=1e-2, rtol=1e-2))
