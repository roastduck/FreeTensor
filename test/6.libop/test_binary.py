import torch
import pytest
import operator
import numpy as np

import freetensor as ft
from freetensor import libop


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.add_, operator.add),
    (libop.sub_, operator.sub),
    (libop.mul_, operator.mul),
    (libop.truediv_, operator.truediv),
])
def test_same_static_shape(libop_func, torch_func):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y, out):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "input", "cpu"]
        out: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: to_test"
        libop_func(x, y, out)

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ft.Array(out_torch.numpy(), device)
    f(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy())

    assert torch.all(torch.isclose(out_torch, torch_func(x_torch, y_torch)))


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.add_, operator.add),
    (libop.sub_, operator.sub),
    (libop.mul_, operator.mul),
    (libop.truediv_, operator.truediv),
])
def test_static_broadcast_shorter(libop_func, torch_func):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y, out):
        x: ft.Var[(4,), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "input", "cpu"]
        out: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: to_test"
        libop_func(x, y, out)

    x_torch = torch.rand(4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ft.Array(out_torch.numpy(), device)
    f(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy())

    assert torch.all(torch.isclose(out_torch, torch_func(x_torch, y_torch)))


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.add_, operator.add),
    (libop.sub_, operator.sub),
    (libop.mul_, operator.mul),
    (libop.truediv_, operator.truediv),
])
def test_static_broadcast_1_at_front(libop_func, torch_func):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y, out):
        x: ft.Var[(1, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "input", "cpu"]
        out: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: out_shape"
        out_shape = ft.empty((2,), "int32", "cpu")
        "nid: to_test"
        libop_func(x, y, out)

    x_torch = torch.rand(1, 4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ft.Array(out_torch.numpy(), device)
    f(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy())

    assert torch.all(torch.isclose(out_torch, torch_func(x_torch, y_torch)))


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.add_, operator.add),
    (libop.sub_, operator.sub),
    (libop.mul_, operator.mul),
    (libop.truediv_, operator.truediv),
])
def test_static_broadcast_1_at_back(libop_func, torch_func):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y, out):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 1), "float32", "input", "cpu"]
        out: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: out_shape"
        out_shape = ft.empty((2,), "int32", "cpu")
        "nid: to_test"
        libop_func(x, y, out)

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 1, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ft.Array(out_torch.numpy(), device)
    f(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy())

    assert torch.all(torch.isclose(out_torch, torch_func(x_torch, y_torch)))


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.add_, operator.add),
    (libop.sub_, operator.sub),
    (libop.mul_, operator.mul),
    (libop.truediv_, operator.truediv),
])
def test_different_dtype(libop_func, torch_func):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y, out):
        x: ft.Var[(4, 4), "int32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "input", "cpu"]
        out: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: out_shape"
        out_shape = ft.empty((2,), "int32", "cpu")
        "nid: to_test"
        libop_func(x, y, out)

    x_torch = torch.randint(0, 100, (4, 4), dtype=torch.int32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ft.Array(out_torch.numpy(), device)
    f(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy())

    assert torch.all(torch.isclose(out_torch, torch_func(x_torch, y_torch)))


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.add, operator.add),
    (libop.sub, operator.sub),
    (libop.mul, operator.mul),
    (libop.truediv, operator.truediv),
])
def test_out_of_place(libop_func, torch_func):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y, out_shape, out):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "input", "cpu"]
        out_shape: ft.Var[(2,), "int32", "output", "cpu"]
        out: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: to_test"
        _out = libop_func(x, y)
        for i in range(2):
            out_shape[i] = _out.shape(i)
        for i in range(4):
            for j in range(4):
                out[i, j] = _out[i, j]

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_shape_torch = torch.zeros(2, dtype=torch.int32)
    out_shape_arr = ft.Array(out_shape_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ft.Array(out_torch.numpy(), device)
    f(x_arr, y_arr, out_shape_arr, out_arr)
    out_shape_numpy = out_shape_arr.numpy()
    out_torch = torch.Tensor(out_arr.numpy())

    assert np.array_equal(out_shape_numpy, [4, 4])
    assert torch.all(torch.isclose(out_torch, torch_func(x_torch, y_torch)))


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.add, operator.add),
    (libop.sub, operator.sub),
    (libop.mul, operator.mul),
    (libop.truediv, operator.truediv),
])
def test_fallback(libop_func, torch_func):
    x = torch.rand(4, 4, dtype=torch.float32)
    y = torch.rand(4, 4, dtype=torch.float32)
    out = libop_func(x, y)
    assert torch.all(torch.isclose(out, torch_func(x, y)))
