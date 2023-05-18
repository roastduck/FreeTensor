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


def test_basic():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, w, y):
        x: ft.Var[(2, 3, 14, 14), "float32", "input", "cpu"]
        w: ft.Var[(8, 3, 3, 3), "float32", "input", "cpu"]
        y: ft.Var[(2, 8, 12, 12), "float32", "output", "cpu"]
        #! label: conv
        libop.conv_(x, w, None, y, auto_pad='VALID')

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ft.array(w_torch)
    y_torch = torch.zeros(2, 8, 12, 12, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, w_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = torch.nn.functional.conv2d(x_torch, w_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, w, b, y):
        x: ft.Var[(2, 3, 14, 14), "float32", "input", "cpu"]
        w: ft.Var[(8, 3, 3, 3), "float32", "input", "cpu"]
        b: ft.Var[(8,), "float32", "input", "cpu"]
        y: ft.Var[(2, 8, 12, 12), "float32", "output", "cpu"]
        #! label: conv
        libop.conv_(x, w, b, y, auto_pad='VALID')

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ft.array(w_torch)
    b_torch = torch.rand(8, dtype=torch.float32)
    b_arr = ft.array(b_torch)
    y_torch = torch.zeros(2, 8, 12, 12, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, w_arr, b_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, bias=b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_same_pad():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, w, y):
        x: ft.Var[(2, 3, 14, 14), "float32", "input", "cpu"]
        w: ft.Var[(8, 3, 3, 3), "float32", "input", "cpu"]
        y: ft.Var[(2, 8, 14, 14), "float32", "output", "cpu"]
        #! label: conv
        libop.conv_(x, w, None, y, kernel_shape=(3, 3), auto_pad='SAME_UPPER')

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ft.array(w_torch)
    y_torch = torch.zeros(2, 8, 14, 14, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, w_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, padding=[1, 1])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_stride():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, w, y):
        x: ft.Var[(2, 3, 14, 14), "float32", "input", "cpu"]
        w: ft.Var[(8, 3, 3, 3), "float32", "input", "cpu"]
        y: ft.Var[(2, 8, 6, 6), "float32", "output", "cpu"]
        #! label: conv
        libop.conv_(x, w, None, y, auto_pad='VALID', strides=(2, 2))

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ft.array(w_torch)
    y_torch = torch.zeros(2, 8, 6, 6, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, w_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, stride=(2, 2))
    assert torch.all(torch.isclose(y_torch, y_std))


def test_group():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, w, y):
        x: ft.Var[(2, 4, 14, 14), "float32", "input", "cpu"]
        w: ft.Var[(8, 2, 3, 3), "float32", "input", "cpu"]
        y: ft.Var[(2, 8, 12, 12), "float32", "output", "cpu"]
        #! label: conv
        libop.conv_(x, w, None, y, auto_pad='VALID', group=2)

    x_torch = torch.rand(2, 4, 14, 14, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    w_torch = torch.rand(8, 2, 3, 3, dtype=torch.float32)
    w_arr = ft.array(w_torch)
    y_torch = torch.zeros(2, 8, 12, 12, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, w_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, groups=2)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_dilation():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, w, y):
        x: ft.Var[(2, 3, 14, 14), "float32", "input", "cpu"]
        w: ft.Var[(8, 3, 3, 3), "float32", "input", "cpu"]
        y: ft.Var[(2, 8, 10, 10), "float32", "output", "cpu"]
        #! label: conv
        libop.conv_(x, w, None, y, auto_pad='VALID', dilations=(2, 2))

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ft.array(w_torch)
    y_torch = torch.zeros(2, 8, 10, 10, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, w_arr, y_arr)
    y_torch = y_arr.torch()

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, dilation=(2, 2))
    assert torch.all(torch.isclose(y_torch, y_std))


def test_out_of_place():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, w):
        x: ft.Var[(2, 3, 14, 14), "float32", "input", "cpu"]
        w: ft.Var[(8, 3, 3, 3), "float32", "input", "cpu"]
        #! label: conv
        return libop.conv(x, w, auto_pad='VALID')

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ft.array(w_torch)
    y_arr = f(x_arr, w_arr)
    y_torch = y_arr.torch()

    y_std = torch.nn.functional.conv2d(x_torch, w_torch)
    assert np.array_equal(y_arr.shape, [2, 8, 12, 12])
    assert torch.all(torch.isclose(y_torch, y_std))
