import freetensor as ft
from freetensor import libop

if not ft.with_pytorch():
    pytest.skip(
        "The tests requires PyTorch, and FreeTensor is expected to be built with "
        "PyTorch to be compatible with it, even if there is no direct interaction "
        "between FreeTensor and PyTorch",
        allow_module_level=True)

import torch


def test_float():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(y):
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        libop.ones_(y)

    y_torch = torch.ones(4, 4, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(y_arr)
    y_torch = y_arr.torch()

    assert torch.all(y_torch == torch.ones(4, 4, dtype=torch.float32))


def test_int():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(y):
        y: ft.Var[(4, 4), "int32", "output", "cpu"]
        libop.ones_(y)

    y_torch = torch.ones(4, 4, dtype=torch.int32)
    y_arr = ft.array(y_torch)
    f(y_arr)
    y_torch = y_arr.torch()

    assert torch.all(y_torch == torch.ones(4, 4, dtype=torch.int32))


def test_bool():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(y):
        y: ft.Var[(4, 4), "bool", "output", "cpu"]
        libop.ones_(y)

    y_torch = torch.ones(4, 4, dtype=torch.bool)
    y_arr = ft.array(y_torch)
    f(y_arr)
    y_torch = y_arr.torch()

    assert torch.all(y_torch == torch.ones(4, 4, dtype=torch.bool))


def test_out_of_place():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f():
        return libop.ones((4, 4), "float32")

    y_arr = f()
    y_torch = y_arr.torch()

    assert torch.all(y_torch == torch.ones(4, 4, dtype=torch.float32))
