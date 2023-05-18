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


def test_static():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(3, 4, 5), "float32", "input", "cpu"]
        y: ft.Var[(60,), "float32", "output", "cpu"]
        #! label: flatten
        libop.flatten_pytorch_(x, y)

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    y_torch = torch.zeros(60, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, y_arr)
    y_torch = y_arr.torch()

    assert torch.all(torch.isclose(y_torch, x_torch.reshape(-1)))


def test_axis():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(3, 4, 5, 6), "float32", "input", "cpu"]
        y: ft.Var[(3, 20, 6), "float32", "output", "cpu"]
        #! label: flatten
        libop.flatten_pytorch_(x, y, start_dim=1, end_dim=2)

    x_torch = torch.rand(3, 4, 5, 6, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    y_torch = torch.zeros(3, 20, 6, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, y_arr)
    y_torch = y_arr.torch()

    assert torch.all(torch.isclose(y_torch, x_torch.reshape(3, -1, 6)))


def test_circular_axis():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(3, 4, 5, 6), "float32", "input", "cpu"]
        y: ft.Var[(3, 20, 6), "float32", "output", "cpu"]
        #! label: flatten
        libop.flatten_pytorch_(x, y, start_dim=1, end_dim=-2)

    x_torch = torch.rand(3, 4, 5, 6, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    y_torch = torch.zeros(3, 20, 6, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, y_arr)
    y_torch = y_arr.torch()

    assert torch.all(torch.isclose(y_torch, x_torch.reshape(3, -1, 6)))


def test_out_of_place():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x):
        x: ft.Var[(3, 4, 5), "float32", "input", "cpu"]
        #! label: flatten
        return libop.flatten_pytorch(x)

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    y_arr = f(x_arr)
    y_torch = y_arr.torch()

    assert np.array_equal(y_arr.shape, [60])
    assert torch.all(torch.isclose(y_torch, x_torch.reshape(-1)))
