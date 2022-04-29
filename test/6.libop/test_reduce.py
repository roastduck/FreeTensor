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

    @ft.transform
    def f(x, y):
        ft.declare_var(x, (3, 4, 5), "float32", "input", "cpu")
        ft.declare_var(y, (3, 5), "float32", "output", "cpu")
        "nid: reduce"
        libop_func(x, y, axes=[1], keepdims=False)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(3, 5, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch, torch_func(x_torch, axis=1)))


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.reduce_sum_, torch.sum),
    (libop.reduce_max_, lambda *args, **kvs: torch.max(*args, **kvs).values),
])
def test_keepdims(libop_func, torch_func):
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        ft.declare_var(x, (3, 4, 5), "float32", "input", "cpu")
        ft.declare_var(y, (3, 1, 5), "float32", "output", "cpu")
        "nid: reduce"
        libop_func(x, y, axes=[1], keepdims=True)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(3, 1, 5, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    assert torch.all(
        torch.isclose(y_torch, torch_func(x_torch, axis=1, keepdim=True)))


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.reduce_sum, torch.sum),
    (libop.reduce_max, lambda *args, **kvs: torch.max(*args, **kvs).values),
])
def test_out_of_place(libop_func, torch_func):
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y_shape, y):
        ft.declare_var(x, (3, 4, 5), "float32", "input", "cpu")
        ft.declare_var(y_shape, (2,), "int32", "output", "cpu")
        ft.declare_var(y, (3, 5), "float32", "output", "cpu")
        "nid: reduce"
        _y = libop_func(x, axes=[1], keepdims=False)
        for i in range(2):
            y_shape[i] = _y.shape(i)
        for i in range(3):
            for j in range(5):
                y[i, j] = _y[i, j]

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_shape_torch = torch.zeros(2, dtype=torch.int32)
    y_shape_arr = ft.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(3, 5, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy())

    assert np.array_equal(y_shape_np, [3, 5])
    assert torch.all(torch.isclose(y_torch, torch_func(x_torch, axis=1)))


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.reduce_sum, torch.sum),
    (libop.reduce_max, lambda *args, **kvs: torch.max(*args, **kvs).values),
])
def test_out_of_place_keepdims(libop_func, torch_func):
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y_shape, y):
        ft.declare_var(x, (3, 4, 5), "float32", "input", "cpu")
        ft.declare_var(y_shape, (3,), "int32", "output", "cpu")
        ft.declare_var(y, (3, 1, 5), "float32", "output", "cpu")
        "nid: reduce"
        _y = libop_func(x, axes=[1], keepdims=True)
        for i in range(3):
            y_shape[i] = _y.shape(i)
        for i in range(3):
            for j in range(5):
                y[i, 0, j] = _y[i, 0, j]

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_shape_torch = torch.zeros(3, dtype=torch.int32)
    y_shape_arr = ft.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(3, 1, 5, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy())

    assert np.array_equal(y_shape_np, [3, 1, 5])
    assert torch.all(
        torch.isclose(y_torch, torch_func(x_torch, axis=1, keepdim=True)))
