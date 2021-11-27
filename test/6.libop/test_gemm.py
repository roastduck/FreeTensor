import torch
import numpy as np

import ir
import ir.libop


def test_basic():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(a, b, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (5, 6), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        ir.libop.gemm_()(a, b, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch @ b_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_trans_A():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(a, b, y):
        ir.declare_var(a, (5, 4), "float32", "input", "cpu")
        ir.declare_var(b, (5, 6), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        ir.libop.gemm_(trans_A=True)(a, b, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(5, 4, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch.t() @ b_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_trans_B():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(a, b, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (6, 5), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        ir.libop.gemm_(trans_B=True)(a, b, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(6, 5, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch @ b_torch.t()
    assert torch.all(torch.isclose(y_torch, y_std))


def test_trans_AB():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(a, b, y):
        ir.declare_var(a, (5, 4), "float32", "input", "cpu")
        ir.declare_var(b, (6, 5), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        ir.libop.gemm_(trans_A=True, trans_B=True)(a, b, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(5, 4, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(6, 5, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch.t() @ b_torch.t()
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(a, b, c, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (5, 6), "float32", "input", "cpu")
        ir.declare_var(c, (4, 6), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        ir.libop.gemm_(has_bias=True)(a, b, c, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    c_torch = torch.rand(4, 6, dtype=torch.float32)
    c_arr = ir.Array(c_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, c_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch @ b_torch + c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias_broadcast_1():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(a, b, c, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (5, 6), "float32", "input", "cpu")
        ir.declare_var(c, (4, 1), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        ir.libop.gemm_(has_bias=True)(a, b, c, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    c_torch = torch.rand(4, 1, dtype=torch.float32)
    c_arr = ir.Array(c_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, c_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch @ b_torch + c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias_broadcast_2():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(a, b, c, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (5, 6), "float32", "input", "cpu")
        ir.declare_var(c, (6,), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        ir.libop.gemm_(has_bias=True)(a, b, c, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    c_torch = torch.rand(6, dtype=torch.float32)
    c_arr = ir.Array(c_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, c_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch @ b_torch + c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias_with_coeff():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(a, b, c, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (5, 6), "float32", "input", "cpu")
        ir.declare_var(c, (4, 6), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        ir.libop.gemm_(has_bias=True, alpha=2.5, beta=3.8)(a, b, c, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    c_torch = torch.rand(4, 6, dtype=torch.float32)
    c_arr = ir.Array(c_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, c_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = 2.5 * a_torch @ b_torch + 3.8 * c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_out_of_place():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(a, b, y_shape, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (5, 6), "float32", "input", "cpu")
        ir.declare_var(y_shape, (2,), "int32", "output", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        _y = ir.libop.gemm()(a, b)
        for i in range(2):
            y_shape[i] = _y.shape(i)
        for i in range(4):
            for j in range(6):
                y[i, j] = _y[i, j]

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    y_shape_torch = torch.zeros(2, dtype=torch.int32)
    y_shape_arr = ir.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch @ b_torch
    assert np.array_equal(y_shape_np, [4, 6])
    assert torch.all(torch.isclose(y_torch, y_std))
