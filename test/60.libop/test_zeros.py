import torch
import numpy as np

import freetensor as ft
from freetensor import libop


def test_float():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(y):
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        libop.zeros_(y)

    y_torch = torch.ones(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy())
    f(y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert torch.all(y_torch == torch.zeros(4, 4, dtype=torch.float32))


def test_int():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(y):
        y: ft.Var[(4, 4), "int32", "output", "cpu"]
        libop.zeros_(y)

    y_torch = torch.ones(4, 4, dtype=torch.int32)
    y_arr = ft.Array(y_torch.numpy())
    f(y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert torch.all(y_torch == torch.zeros(4, 4, dtype=torch.int32))


def test_bool():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(y):
        y: ft.Var[(4, 4), "bool", "output", "cpu"]
        libop.zeros_(y)

    y_torch = torch.ones(4, 4, dtype=torch.bool)
    y_arr = ft.Array(y_torch.numpy())
    f(y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert torch.all(y_torch == torch.zeros(4, 4, dtype=torch.bool))


def test_out_of_place():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f():
        return libop.zeros((4, 4), "float32")

    y_arr = f()
    y_torch = torch.tensor(y_arr.numpy())

    assert torch.all(y_torch == torch.zeros(4, 4, dtype=torch.float32))
