import freetensor as ft
from freetensor import debug
import pytest
import numpy as np

if not ft.with_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

device = ft.GPU()
target = device.target()


def test_basic():

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
    s.as_matmul("L1", ft.AsMatMulMode.KeepMemLayout, target, "cublas")
    func = ft.lower(s.func(), target, verbose=1)
    code = ft.codegen(func, target, verbose=True)
    assert "cublas" in code.code
    a_np = np.random.uniform(size=(48, 64)).astype("float32")
    b_np = np.random.uniform(size=(64, 72)).astype("float32")
    c_np = np.random.uniform(size=(48, 72)).astype("float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np.copy())
    ft.build_binary(code, device)(a=a_arr, b=b_arr, c=c_arr)
    c_result = c_arr.numpy()

    assert np.all(np.isclose(c_result, c_np + a_np @ b_np))
