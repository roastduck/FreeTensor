import torch
import numpy as np

import freetensor as ft
from freetensor import libop


def test_static():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        x: ft.Var((3, 4, 5), "float32", "input", "cpu")
        y: ft.Var((3, 20), "float32", "output", "cpu")
        "nid: flatten"
        libop.flatten_(x, y)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(3, 20, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch, x_torch.reshape(3, -1)))


def test_axis():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        x: ft.Var((3, 4, 5), "float32", "input", "cpu")
        y: ft.Var((12, 5), "float32", "output", "cpu")
        "nid: flatten"
        libop.flatten_(x, y, axis=2)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(12, 5, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch, x_torch.reshape(-1, 5)))


def test_out_of_place():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y_shape, y):
        x: ft.Var((3, 4, 5), "float32", "input", "cpu")
        y_shape: ft.Var((2,), "int32", "output", "cpu")
        y: ft.Var((3, 20), "float32", "output", "cpu")
        "nid: flatten"
        _y = libop.flatten(x)
        for i in range(2):
            y_shape[i] = _y.shape(i)
        for i in range(3):
            for j in range(20):
                y[i, j] = _y[i, j]

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_shape_torch = torch.zeros(2, dtype=torch.int32)
    y_shape_arr = ft.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(3, 20, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy())

    assert np.array_equal(y_shape_np, [3, 20])
    assert torch.all(torch.isclose(y_torch, x_torch.reshape(3, -1)))
