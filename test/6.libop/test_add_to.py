import torch
import numpy as np

import freetensor as ft
from freetensor import libop


def test_same_static_shape():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        x: ft.Var((4, 4), "float32", "input", "cpu")
        y: ft.Var((4, 4), "float32", "output", "cpu")
        "nid: add_to"
        libop.add_to(y, x)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch_new = torch.Tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch_new, x_torch + y_torch))


def test_static_broadcast_shorter():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        x: ft.Var((4,), "float32", "input", "cpu")
        y: ft.Var((4, 4), "float32", "output", "cpu")
        "nid: add_to"
        libop.add_to(y, x)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch_new = torch.Tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch_new, x_torch + y_torch))


def test_static_broadcast_1_at_front():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        x: ft.Var((1, 4), "float32", "input", "cpu")
        y: ft.Var((4, 4), "float32", "output", "cpu")
        "nid: add_to"
        libop.add_to(y, x)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(1, 4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch_new = torch.Tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch_new, x_torch + y_torch))


def test_static_broadcast_1_at_back():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        x: ft.Var((4, 1), "float32", "input", "cpu")
        y: ft.Var((4, 4), "float32", "output", "cpu")
        "nid: add_to"
        libop.add_to(y, x)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(4, 1, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch_new = torch.Tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch_new, x_torch + y_torch))


def test_different_dtype():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        x: ft.Var((4, 4), "int32", "input", "cpu")
        y: ft.Var((4, 4), "float32", "output", "cpu")
        "nid: add_to"
        libop.add_to(y, x)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.randint(0, 100, (4, 4), dtype=torch.int32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch_new = torch.Tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch_new, x_torch + y_torch))
