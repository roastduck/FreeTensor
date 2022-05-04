import torch
import numpy as np

import freetensor as ft
from freetensor import libop


def test_basic():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(a, b, y):
        a: ft.Var((4, 5), "float32", "input", "cpu")
        b: ft.Var((5,), "float32", "input", "cpu")
        y: ft.Var((4,), "float32", "output", "cpu")
        "nid: einsum"
        libop.einsum_("ij,j->i", a, b, y)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.einsum("ij,j->i", a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_broadcast():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(a, b, y):
        a: ft.Var((4, 1), "float32", "input", "cpu")
        b: ft.Var((5,), "float32", "input", "cpu")
        y: ft.Var((4,), "float32", "output", "cpu")
        "nid: einsum"
        libop.einsum_("ij,j->i", a, b, y)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    a_torch = torch.rand(4, 1, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.einsum("ij,j->i", a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_out_of_place():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(a, b, y_shape, y):
        a: ft.Var((4, 5), "float32", "input", "cpu")
        b: ft.Var((5,), "float32", "input", "cpu")
        y_shape: ft.Var((1,), "int32", "output", "cpu")
        y: ft.Var((4,), "float32", "output", "cpu")
        "nid: gemm"
        _y = libop.einsum("ij,j->i", a, b)
        y_shape[0] = _y.shape(0)
        for i in range(4):
            y[i] = _y[i]

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy(), device)
    y_shape_torch = torch.zeros(1, dtype=torch.int32)
    y_shape_arr = ft.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(a_arr, b_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.einsum("ij,j->i", a_torch, b_torch)
    assert np.array_equal(y_shape_np, [4])
    assert torch.all(torch.isclose(y_torch, y_std))
