import ir
import ir.debug
import pytest
import numpy as np

target = ir.GPU()
device = ir.Device(target)


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
    ir.Driver(func, code, device)(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))
