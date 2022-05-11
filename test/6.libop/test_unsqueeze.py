import torch
import numpy as np

import freetensor as ft
from freetensor import libop


def test_static():
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(3, 4, 5), "float32", "input", "cpu"]
        y: ft.Var[(3, 1, 4, 1, 5), "float32", "output", "cpu"]
        "nid: unsqueeze"
        libop.unsqueeze_(x, y, axes=[1, 3])

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(3, 1, 4, 1, 5, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    f(x_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch, x_torch.reshape(3, 1, 4, 1, 5)))


def test_out_of_place():
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y_shape, y):
        x: ft.Var[(3, 4, 5), "float32", "input", "cpu"]
        y_shape: ft.Var[(5,), "int32", "output", "cpu"]
        y: ft.Var[(3, 1, 4, 1, 5), "float32", "output", "cpu"]
        "nid: unsqueeze"
        _y = libop.unsqueeze(x, axes=[1, 3])
        for i in range(5):
            y_shape[i] = _y.shape(i)
        for i in range(3):
            for j in range(4):
                for k in range(5):
                    y[i, 0, j, 0, k] = _y[i, 0, j, 0, k]

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_shape_torch = torch.zeros(5, dtype=torch.int32)
    y_shape_arr = ft.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(3, 1, 4, 1, 5, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    f(x_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.tensor(y_arr.numpy())

    assert np.array_equal(y_shape_np, [3, 1, 4, 1, 5])
    assert torch.all(torch.isclose(y_torch, x_torch.reshape(3, 1, 4, 1, 5)))
