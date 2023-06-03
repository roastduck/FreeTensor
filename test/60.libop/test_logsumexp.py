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


def test_static_shape():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4,), "float32", "output", "cpu"]
        #! label: logsumexp
        libop.logsumexp_(x, y, axis=-1, keepdims=False)

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    y_torch = torch.zeros(4, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, y_arr)
    y_torch = y_arr.torch()

    assert torch.all(torch.isclose(y_torch, torch.logsumexp(x_torch, axis=-1)))


def test_out_of_place():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        #! label: logsumexp
        return libop.logsumexp(x, axis=-1, keepdims=False)

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    y_arr = f(x_arr)
    y_torch = y_arr.torch()

    assert np.array_equal(y_arr.shape, [4])
    assert torch.all(torch.isclose(y_torch, torch.logsumexp(x_torch, axis=-1)))


def test_grad():
    device = ft.CPU()

    @ft.transform
    def f(x, y):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4,), "float32", "output", "cpu"]
        #! label: logsumexp
        libop.logsumexp_(x, y, axis=-1, keepdims=False)

    print(f)
    f, g, requires, provides = ft.grad_(f, ["x"], ["y"],
                                        ft.GradTapeMode.NoReuseOnly)
    print("Forward:")
    f = ft.optimize(f, verbose=1)
    print("Backward:")
    g = ft.optimize(g, verbose=1)

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    x_torch.requires_grad = True
    y_torch_ours = torch.zeros(4, dtype=torch.float32)
    y_arr = ft.array(y_torch_ours)
    f(x_arr, y_arr)
    y_torch_ours = y_arr.torch()
    y_torch = torch.logsumexp(x_torch, axis=-1)
    assert torch.all(torch.isclose(y_torch_ours, y_torch))

    y_torch.grad = torch.rand(4, dtype=torch.float32)
    d_y_arr = ft.array(y_torch.grad.clone())
    x_grad_torch_ours = torch.zeros(4, 4, dtype=torch.float32)
    d_x_arr = ft.array(x_grad_torch_ours)
    g(**{provides['y']: d_y_arr, requires['x']: d_x_arr})
    x_grad_torch_ours = d_x_arr.torch()
    y_torch.backward(y_torch.grad)
    assert torch.all(torch.isclose(x_grad_torch_ours, x_torch.grad, 1e-4, 1e-7))
