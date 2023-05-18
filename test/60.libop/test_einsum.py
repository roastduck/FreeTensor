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
        b: ft.Var[(5,), "float32", "input", "cpu"]
        y: ft.Var[(4,), "float32", "output", "cpu"]
        #! label: einsum
        libop.einsum_("ij,j->i", a, b, y)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.array(a_torch)
    b_torch = torch.rand(5, dtype=torch.float32)
    b_arr = ft.array(b_torch)
    y_torch = torch.zeros(4, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(a_arr, b_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = torch.einsum("ij,j->i", a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_broadcast():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, y):
        a: ft.Var[(4, 1), "float32", "input", "cpu"]
        b: ft.Var[(5,), "float32", "input", "cpu"]
        y: ft.Var[(4,), "float32", "output", "cpu"]
        #! label: einsum
        libop.einsum_("ij,j->i", a, b, y)

    a_torch = torch.rand(4, 1, dtype=torch.float32)
    a_arr = ft.array(a_torch)
    b_torch = torch.rand(5, dtype=torch.float32)
    b_arr = ft.array(b_torch)
    y_torch = torch.zeros(4, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(a_arr, b_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = torch.einsum("ij,j->i", a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_out_of_place():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5,), "float32", "input", "cpu"]
        #! label: gemm
        return libop.einsum("ij,j->i", a, b)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.array(a_torch)
    b_torch = torch.rand(5, dtype=torch.float32)
    b_arr = ft.array(b_torch)
    y_arr = f(a_arr, b_arr)
    y_torch = y_arr.torch()

    y_std = torch.einsum("ij,j->i", a_torch, b_torch)
    assert np.array_equal(y_arr.shape, [4])
    assert torch.all(torch.isclose(y_torch, y_std))
