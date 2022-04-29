import torch
import numpy as np

import freetensor as ft
from freetensor import libop


def test_basic():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(a, b, y):
        ft.declare_var(a, (4, 5), "float32", "input", "cpu")
        ft.declare_var(b, (5, 6), "float32", "input", "cpu")
        ft.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        libop.gemm_(a, b, None, y)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch @ b_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_trans_A():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(a, b, y):
        ft.declare_var(a, (5, 4), "float32", "input", "cpu")
        ft.declare_var(b, (5, 6), "float32", "input", "cpu")
        ft.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        libop.gemm_(a, b, None, y, trans_A=True)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    a_torch = torch.rand(5, 4, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch.t() @ b_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_trans_B():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(a, b, y):
        ft.declare_var(a, (4, 5), "float32", "input", "cpu")
        ft.declare_var(b, (6, 5), "float32", "input", "cpu")
        ft.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        libop.gemm_(a, b, None, y, trans_B=True)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy(), device)
    b_torch = torch.rand(6, 5, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch @ b_torch.t()
    assert torch.all(torch.isclose(y_torch, y_std))


def test_trans_AB():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(a, b, y):
        ft.declare_var(a, (5, 4), "float32", "input", "cpu")
        ft.declare_var(b, (6, 5), "float32", "input", "cpu")
        ft.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        libop.gemm_(a, b, None, y, trans_A=True, trans_B=True)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    a_torch = torch.rand(5, 4, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy(), device)
    b_torch = torch.rand(6, 5, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch.t() @ b_torch.t()
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(a, b, c, y):
        ft.declare_var(a, (4, 5), "float32", "input", "cpu")
        ft.declare_var(b, (5, 6), "float32", "input", "cpu")
        ft.declare_var(c, (4, 6), "float32", "input", "cpu")
        ft.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        libop.gemm_(a, b, c, y)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy(), device)
    c_torch = torch.rand(4, 6, dtype=torch.float32)
    c_arr = ft.Array(c_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(a_arr, b_arr, c_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch @ b_torch + c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias_broadcast_1():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(a, b, c, y):
        ft.declare_var(a, (4, 5), "float32", "input", "cpu")
        ft.declare_var(b, (5, 6), "float32", "input", "cpu")
        ft.declare_var(c, (4, 1), "float32", "input", "cpu")
        ft.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        libop.gemm_(a, b, c, y)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy(), device)
    c_torch = torch.rand(4, 1, dtype=torch.float32)
    c_arr = ft.Array(c_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(a_arr, b_arr, c_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch @ b_torch + c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias_broadcast_2():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(a, b, c, y):
        ft.declare_var(a, (4, 5), "float32", "input", "cpu")
        ft.declare_var(b, (5, 6), "float32", "input", "cpu")
        ft.declare_var(c, (6,), "float32", "input", "cpu")
        ft.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        libop.gemm_(a, b, c, y)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy(), device)
    c_torch = torch.rand(6, dtype=torch.float32)
    c_arr = ft.Array(c_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(a_arr, b_arr, c_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch @ b_torch + c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias_with_coeff():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(a, b, c, y):
        ft.declare_var(a, (4, 5), "float32", "input", "cpu")
        ft.declare_var(b, (5, 6), "float32", "input", "cpu")
        ft.declare_var(c, (4, 6), "float32", "input", "cpu")
        ft.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        libop.gemm_(a, b, c, y, alpha=2.5, beta=3.8)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy(), device)
    c_torch = torch.rand(4, 6, dtype=torch.float32)
    c_arr = ft.Array(c_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(a_arr, b_arr, c_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = 2.5 * a_torch @ b_torch + 3.8 * c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_out_of_place():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(a, b, y_shape, y):
        ft.declare_var(a, (4, 5), "float32", "input", "cpu")
        ft.declare_var(b, (5, 6), "float32", "input", "cpu")
        ft.declare_var(y_shape, (2,), "int32", "output", "cpu")
        ft.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: gemm"
        _y = libop.gemm(a, b)
        for i in range(2):
            y_shape[i] = _y.shape(i)
        for i in range(4):
            for j in range(6):
                y[i, j] = _y[i, j]

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy(), device)
    y_shape_torch = torch.zeros(2, dtype=torch.int32)
    y_shape_arr = ft.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(a_arr, b_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = a_torch @ b_torch
    assert np.array_equal(y_shape_np, [4, 6])
    assert torch.all(torch.isclose(y_torch, y_std))
