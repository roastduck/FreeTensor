import freetensor as ft
from freetensor import libop

if not ft.with_pytorch():
    pytest.skip(
        "The tests requires PyTorch, and FreeTensor is expected to be built with "
        "PyTorch to be compatible with it, even if there is no direct interaction "
        "between FreeTensor and PyTorch",
        allow_module_level=True)

import torch


def test_same_static_shape():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        #! label: assign
        libop.assign(y, x)

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, y_arr)
    y_torch = y_arr.torch()

    assert torch.all(torch.isclose(y_torch, x_torch))


def test_static_broadcast_shorter():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4,), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        #! label: assign
        libop.assign(y, x)

    x_torch = torch.rand(4, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, y_arr)
    y_torch = y_arr.torch()

    assert torch.all(torch.isclose(y_torch, x_torch))


def test_static_broadcast_1_at_front():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(1, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        #! label: assign
        libop.assign(y, x)

    x_torch = torch.rand(1, 4, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, y_arr)
    y_torch = y_arr.torch()

    assert torch.all(torch.isclose(y_torch, x_torch))


def test_static_broadcast_1_at_back():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4, 1), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        #! label: assign
        libop.assign(y, x)

    x_torch = torch.rand(4, 1, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, y_arr)
    y_torch = y_arr.torch()

    assert torch.all(torch.isclose(y_torch, x_torch))


def test_different_dtype():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4, 4), "int32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        #! label: assign
        libop.assign(y, x)

    x_torch = torch.randint(0, 100, (4, 4), dtype=torch.int32)
    x_arr = ft.array(x_torch)
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, y_arr)
    y_torch = y_arr.torch()

    assert torch.all(torch.isclose(y_torch, x_torch.float()))


def test_operator_overload():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        #! label: assign
        y[:] = x

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.array(x_torch)
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ft.array(y_torch)
    f(x_arr, y_arr)
    y_torch = y_arr.torch()

    assert torch.all(torch.isclose(y_torch, x_torch))
