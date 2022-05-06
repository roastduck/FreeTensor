import torch
import numpy as np

import freetensor as ft
from freetensor import libop


def test_same_static_shape():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y, out):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "input", "cpu"]
        out: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: add"
        libop.add_(x, y, out)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ft.Array(out_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy())

    assert torch.all(torch.isclose(out_torch, x_torch + y_torch))


def test_static_broadcast_shorter():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y, out):
        x: ft.Var[(4,), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "input", "cpu"]
        out: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: add"
        libop.add_(x, y, out)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ft.Array(out_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy())

    assert torch.all(torch.isclose(out_torch, x_torch + y_torch))


def test_static_broadcast_1_at_front():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y, out):
        x: ft.Var[(1, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "input", "cpu"]
        out: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: out_shape"
        out_shape = ft.empty((2,), "int32", "cpu")
        "nid: add"
        libop.add_(x, y, out)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(1, 4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ft.Array(out_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy())

    assert torch.all(torch.isclose(out_torch, x_torch + y_torch))


def test_static_broadcast_1_at_back():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y, out):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 1), "float32", "input", "cpu"]
        out: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: out_shape"
        out_shape = ft.empty((2,), "int32", "cpu")
        "nid: add"
        libop.add_(x, y, out)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 1, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ft.Array(out_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy())

    assert torch.all(torch.isclose(out_torch, x_torch + y_torch))


def test_different_dtype():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y, out):
        x: ft.Var[(4, 4), "int32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "input", "cpu"]
        out: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: out_shape"
        out_shape = ft.empty((2,), "int32", "cpu")
        "nid: add"
        libop.add_(x, y, out)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.randint(0, 100, (4, 4), dtype=torch.int32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ft.Array(out_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy())

    assert torch.all(torch.isclose(out_torch, x_torch + y_torch))


def test_out_of_place():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y, out_shape, out):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "input", "cpu"]
        out_shape: ft.Var[(2,), "int32", "output", "cpu"]
        out: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: add"
        _out = libop.add(x, y)
        for i in range(2):
            out_shape[i] = _out.shape(i)
        for i in range(4):
            for j in range(4):
                out[i, j] = _out[i, j]

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_shape_torch = torch.zeros(2, dtype=torch.int32)
    out_shape_arr = ft.Array(out_shape_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ft.Array(out_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr, out_shape_arr, out_arr)
    out_shape_numpy = out_shape_arr.numpy()
    out_torch = torch.Tensor(out_arr.numpy())

    assert np.array_equal(out_shape_numpy, [4, 4])
    assert torch.all(torch.isclose(out_torch, x_torch + y_torch))


def test_fallback():
    x = torch.rand(4, 4, dtype=torch.float32)
    y = torch.rand(4, 4, dtype=torch.float32)
    out = libop.add(x, y)
    assert torch.all(torch.isclose(out, x + y))
