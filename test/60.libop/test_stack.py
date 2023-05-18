import freetensor as ft
from freetensor import libop

import numpy as np
import pytest

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
    def f(x1, x2, x3, y):
        x1: ft.Var[(3, 4), "float32", "input", "cpu"]
        x2: ft.Var[(3, 4), "float32", "input", "cpu"]
        x3: ft.Var[(3, 4), "float32", "input", "cpu"]
        y: ft.Var[(3, 3, 4), "float32", "output", "cpu"]
        #! label: stack
        libop.stack_([x1, x2, x3], y, axis=1)

    x1_torch = torch.rand(3, 4, dtype=torch.float32)
    x1_arr = ft.array(x1_torch)
    x2_torch = torch.rand(3, 4, dtype=torch.float32)
    x2_arr = ft.array(x2_torch)
    x3_torch = torch.rand(3, 4, dtype=torch.float32)
    x3_arr = ft.array(x3_torch)
    y_torch = torch.zeros(3, 3, 4, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x1_arr, x2_arr, x3_arr, y_arr)
    y_torch = y_arr.torch()

    assert torch.all(
        torch.isclose(y_torch, torch.stack([x1_torch, x2_torch, x3_torch],
                                           dim=1)))


def test_out_of_place():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x1, x2, x3):
        x1: ft.Var[(3, 4), "float32", "input", "cpu"]
        x2: ft.Var[(3, 4), "float32", "input", "cpu"]
        x3: ft.Var[(3, 4), "float32", "input", "cpu"]
        #! label: stack
        return libop.stack([x1, x2, x3], axis=1)

    x1_torch = torch.rand(3, 4, dtype=torch.float32)
    x1_arr = ft.array(x1_torch)
    x2_torch = torch.rand(3, 4, dtype=torch.float32)
    x2_arr = ft.array(x2_torch)
    x3_torch = torch.rand(3, 4, dtype=torch.float32)
    x3_arr = ft.array(x3_torch)
    y_arr = f(x1_arr, x2_arr, x3_arr)
    y_torch = y_arr.torch()

    assert np.array_equal(y_arr.shape, [3, 3, 4])
    assert torch.all(
        torch.isclose(y_torch, torch.stack([x1_torch, x2_torch, x3_torch],
                                           dim=1)))
