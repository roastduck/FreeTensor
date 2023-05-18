import numpy as np

import freetensor as ft
from freetensor import libop

if not ft.with_pytorch():
    pytest.skip(
        "The tests requires PyTorch, and FreeTensor is expected to be built with "
        "PyTorch to be compatible with it, even if there is no direct interaction "
        "between FreeTensor and PyTorch",
        allow_module_level=True)

import torch


def test_basic():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, y):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        y: ft.Var[(4, 6), "float32", "output", "cpu"]
        #! label: gemm
        libop.gemm_(a, b, None, y)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.array(a_torch)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.array(b_torch)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(a_arr, b_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = a_torch @ b_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_trans_A():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, y):
        a: ft.Var[(5, 4), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        y: ft.Var[(4, 6), "float32", "output", "cpu"]
        #! label: gemm
        libop.gemm_(a, b, None, y, trans_A=True)

    a_torch = torch.rand(5, 4, dtype=torch.float32)
    a_arr = ft.array(a_torch)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.array(b_torch)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(a_arr, b_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = a_torch.t() @ b_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_trans_B():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, y):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(6, 5), "float32", "input", "cpu"]
        y: ft.Var[(4, 6), "float32", "output", "cpu"]
        #! label: gemm
        libop.gemm_(a, b, None, y, trans_B=True)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.array(a_torch)
    b_torch = torch.rand(6, 5, dtype=torch.float32)
    b_arr = ft.array(b_torch)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(a_arr, b_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = a_torch @ b_torch.t()
    assert torch.all(torch.isclose(y_torch, y_std))


def test_trans_AB():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, y):
        a: ft.Var[(5, 4), "float32", "input", "cpu"]
        b: ft.Var[(6, 5), "float32", "input", "cpu"]
        y: ft.Var[(4, 6), "float32", "output", "cpu"]
        #! label: gemm
        libop.gemm_(a, b, None, y, trans_A=True, trans_B=True)

    a_torch = torch.rand(5, 4, dtype=torch.float32)
    a_arr = ft.array(a_torch)
    b_torch = torch.rand(6, 5, dtype=torch.float32)
    b_arr = ft.array(b_torch)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(a_arr, b_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = a_torch.t() @ b_torch.t()
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, c, y):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        c: ft.Var[(4, 6), "float32", "input", "cpu"]
        y: ft.Var[(4, 6), "float32", "output", "cpu"]
        #! label: gemm
        libop.gemm_(a, b, c, y)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.array(a_torch)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.array(b_torch)
    c_torch = torch.rand(4, 6, dtype=torch.float32)
    c_arr = ft.array(c_torch)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(a_arr, b_arr, c_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = a_torch @ b_torch + c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias_broadcast_1():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, c, y):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        c: ft.Var[(4, 1), "float32", "input", "cpu"]
        y: ft.Var[(4, 6), "float32", "output", "cpu"]
        #! label: gemm
        libop.gemm_(a, b, c, y)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.array(a_torch)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.array(b_torch)
    c_torch = torch.rand(4, 1, dtype=torch.float32)
    c_arr = ft.array(c_torch)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(a_arr, b_arr, c_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = a_torch @ b_torch + c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias_broadcast_2():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, c, y):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        c: ft.Var[(6,), "float32", "input", "cpu"]
        y: ft.Var[(4, 6), "float32", "output", "cpu"]
        #! label: gemm
        libop.gemm_(a, b, c, y)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.array(a_torch)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.array(b_torch)
    c_torch = torch.rand(6, dtype=torch.float32)
    c_arr = ft.array(c_torch)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(a_arr, b_arr, c_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = a_torch @ b_torch + c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias_with_coeff():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, c, y):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        c: ft.Var[(4, 6), "float32", "input", "cpu"]
        y: ft.Var[(4, 6), "float32", "output", "cpu"]
        #! label: gemm
        libop.gemm_(a, b, c, y, alpha=2.5, beta=3.8)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.array(a_torch)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.array(b_torch)
    c_torch = torch.rand(4, 6, dtype=torch.float32)
    c_arr = ft.array(c_torch)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(a_arr, b_arr, c_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = 2.5 * a_torch @ b_torch + 3.8 * c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_out_of_place():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        #! label: gemm
        return libop.gemm(a, b)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.array(a_torch)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.array(b_torch)
    y_arr = f(a_arr, b_arr)
    y_torch = y_arr.torch()

    y_std = a_torch @ b_torch
    assert np.array_equal(y_arr.shape, [4, 6])
    assert torch.all(torch.isclose(y_torch, y_std))
