import torch
import pytest
import numpy as np

import freetensor as ft
from freetensor import libop


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.reduce_sum_, torch.sum),
    (libop.reduce_max_, lambda *args, **kvs: torch.max(*args, **kvs).values),
])
def test_static(libop_func, torch_func):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(3, 4, 5), "float32", "input", "cpu"]
        y: ft.Var[(3, 5), "float32", "output", "cpu"]
        "nid: reduce"
        libop_func(x, y, axes=[1], keepdims=False)

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(3, 5, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    f(x_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch, torch_func(x_torch, axis=1)))


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.reduce_sum_, torch.sum),
    (libop.reduce_max_, lambda *args, **kvs: torch.max(*args, **kvs).values),
])
def test_keepdims(libop_func, torch_func):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(3, 4, 5), "float32", "input", "cpu"]
        y: ft.Var[(3, 1, 5), "float32", "output", "cpu"]
        "nid: reduce"
        libop_func(x, y, axes=[1], keepdims=True)

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(3, 1, 5, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    f(x_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert torch.all(
        torch.isclose(y_torch, torch_func(x_torch, axis=1, keepdim=True)))


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.reduce_sum, torch.sum),
    (libop.reduce_max, lambda *args, **kvs: torch.max(*args, **kvs).values),
])
def test_out_of_place(libop_func, torch_func):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x):
        x: ft.Var[(3, 4, 5), "float32", "input", "cpu"]
        "nid: reduce"
        return libop_func(x, axes=[1], keepdims=False)

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_arr = f(x_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert np.array_equal(y_arr.shape, [3, 5])
    assert torch.all(torch.isclose(y_torch, torch_func(x_torch, axis=1)))


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.reduce_sum, torch.sum),
    (libop.reduce_max, lambda *args, **kvs: torch.max(*args, **kvs).values),
])
def test_out_of_place_keepdims(libop_func, torch_func):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x):
        x: ft.Var[(3, 4, 5), "float32", "input", "cpu"]
        "nid: reduce"
        return libop_func(x, axes=[1], keepdims=True)

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_arr = f(x_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert np.array_equal(y_arr.shape, [3, 1, 5])
    assert torch.all(
        torch.isclose(y_torch, torch_func(x_torch, axis=1, keepdim=True)))
