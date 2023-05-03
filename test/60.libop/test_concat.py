import freetensor as ft
from freetensor import libop

import torch
import numpy as np


def test_static():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x1, x2, x3, y):
        x1: ft.Var[(3, 1, 4), "float32", "input", "cpu"]
        x2: ft.Var[(3, 2, 4), "float32", "input", "cpu"]
        x3: ft.Var[(3, 3, 4), "float32", "input", "cpu"]
        y: ft.Var[(3, 6, 4), "float32", "output", "cpu"]
        #! label: concat
        libop.concat_([x1, x2, x3], y, axis=1)

    x1_torch = torch.rand(3, 1, 4, dtype=torch.float32)
    x1_arr = ft.Array(x1_torch.numpy())
    x2_torch = torch.rand(3, 2, 4, dtype=torch.float32)
    x2_arr = ft.Array(x2_torch.numpy())
    x3_torch = torch.rand(3, 3, 4, dtype=torch.float32)
    x3_arr = ft.Array(x3_torch.numpy())
    y_torch = torch.zeros(3, 6, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy())
    f(x1_arr, x2_arr, x3_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert torch.all(
        torch.isclose(y_torch, torch.cat([x1_torch, x2_torch, x3_torch],
                                         dim=1)))


def test_out_of_place():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x1, x2, x3):
        x1: ft.Var[(3, 1, 4), "float32", "input", "cpu"]
        x2: ft.Var[(3, 2, 4), "float32", "input", "cpu"]
        x3: ft.Var[(3, 3, 4), "float32", "input", "cpu"]
        #! label: concat
        return libop.concat([x1, x2, x3], axis=1)

    x1_torch = torch.rand(3, 1, 4, dtype=torch.float32)
    x1_arr = ft.Array(x1_torch.numpy())
    x2_torch = torch.rand(3, 2, 4, dtype=torch.float32)
    x2_arr = ft.Array(x2_torch.numpy())
    x3_torch = torch.rand(3, 3, 4, dtype=torch.float32)
    x3_arr = ft.Array(x3_torch.numpy())
    y_arr = f(x1_arr, x2_arr, x3_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert np.array_equal(y_arr.shape, [3, 6, 4])
    assert torch.all(
        torch.isclose(y_torch, torch.cat([x1_torch, x2_torch, x3_torch],
                                         dim=1)))
