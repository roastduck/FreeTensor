import numpy as np
import pytest

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
        y: ft.Var[(3, 1, 4, 1, 5), "float32", "output", "cpu"]
        #! label: unsqueeze
        libop.unsqueeze_(x, y, axes=[1, 3])

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    y_torch = torch.zeros(3, 1, 4, 1, 5, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, y_arr)
    y_torch = y_arr.torch()

    assert torch.all(torch.isclose(y_torch, x_torch.reshape(3, 1, 4, 1, 5)))


def test_out_of_place():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x):
        x: ft.Var[(3, 4, 5), "float32", "input", "cpu"]
        #! label: unsqueeze
        return libop.unsqueeze(x, axes=[1, 3])

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    y_arr = f(x_arr)
    y_torch = y_arr.torch()

    assert np.array_equal(y_arr.shape, [3, 1, 4, 1, 5])
    assert torch.all(torch.isclose(y_torch, x_torch.reshape(3, 1, 4, 1, 5)))
