import torch
import numpy as np

import freetensor as ft
from freetensor import libop


def test_basic():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, w, y):
        x: ft.Var((2, 3, 14, 14), "float32", "input", "cpu")
        w: ft.Var((8, 3, 3, 3), "float32", "input", "cpu")
        y: ft.Var((2, 8, 12, 12), "float32", "output", "cpu")
        "nid: conv"
        libop.conv_(x, w, None, y, auto_pad='VALID')

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ft.Array(w_torch.numpy(), device)
    y_torch = torch.zeros(2, 8, 12, 12, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, w_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.nn.functional.conv2d(x_torch, w_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, w, b, y):
        x: ft.Var((2, 3, 14, 14), "float32", "input", "cpu")
        w: ft.Var((8, 3, 3, 3), "float32", "input", "cpu")
        b: ft.Var((8,), "float32", "input", "cpu")
        y: ft.Var((2, 8, 12, 12), "float32", "output", "cpu")
        "nid: conv"
        libop.conv_(x, w, b, y, auto_pad='VALID')

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ft.Array(w_torch.numpy(), device)
    b_torch = torch.rand(8, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(2, 8, 12, 12, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, w_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, bias=b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_same_pad():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, w, y):
        x: ft.Var((2, 3, 14, 14), "float32", "input", "cpu")
        w: ft.Var((8, 3, 3, 3), "float32", "input", "cpu")
        y: ft.Var((2, 8, 14, 14), "float32", "output", "cpu")
        "nid: conv"
        libop.conv_(x, w, None, y, kernel_shape=(3, 3), auto_pad='SAME_UPPER')

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ft.Array(w_torch.numpy(), device)
    y_torch = torch.zeros(2, 8, 14, 14, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, w_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, padding=[1, 1])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_stride():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, w, y):
        x: ft.Var((2, 3, 14, 14), "float32", "input", "cpu")
        w: ft.Var((8, 3, 3, 3), "float32", "input", "cpu")
        y: ft.Var((2, 8, 6, 6), "float32", "output", "cpu")
        "nid: conv"
        libop.conv_(x, w, None, y, auto_pad='VALID', strides=(2, 2))

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ft.Array(w_torch.numpy(), device)
    y_torch = torch.zeros(2, 8, 6, 6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, w_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, stride=(2, 2))
    assert torch.all(torch.isclose(y_torch, y_std))


def test_group():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, w, y):
        x: ft.Var((2, 4, 14, 14), "float32", "input", "cpu")
        w: ft.Var((8, 2, 3, 3), "float32", "input", "cpu")
        y: ft.Var((2, 8, 12, 12), "float32", "output", "cpu")
        "nid: conv"
        libop.conv_(x, w, None, y, auto_pad='VALID', group=2)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(2, 4, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    w_torch = torch.rand(8, 2, 3, 3, dtype=torch.float32)
    w_arr = ft.Array(w_torch.numpy(), device)
    y_torch = torch.zeros(2, 8, 12, 12, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, w_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, groups=2)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_dilation():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, w, y):
        x: ft.Var((2, 3, 14, 14), "float32", "input", "cpu")
        w: ft.Var((8, 3, 3, 3), "float32", "input", "cpu")
        y: ft.Var((2, 8, 10, 10), "float32", "output", "cpu")
        "nid: conv"
        libop.conv_(x, w, None, y, auto_pad='VALID', dilations=(2, 2))

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ft.Array(w_torch.numpy(), device)
    y_torch = torch.zeros(2, 8, 10, 10, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, w_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, dilation=(2, 2))
    assert torch.all(torch.isclose(y_torch, y_std))


def test_out_of_place():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, w, y_shape, y):
        x: ft.Var((2, 3, 14, 14), "float32", "input", "cpu")
        w: ft.Var((8, 3, 3, 3), "float32", "input", "cpu")
        y_shape: ft.Var((4,), "int32", "output", "cpu")
        y: ft.Var((2, 8, 12, 12), "float32", "output", "cpu")
        "nid: conv"
        _y = libop.conv(x, w, auto_pad='VALID')
        for i in range(4):
            y_shape[i] = _y.shape(i)
        for n in range(2):
            for c in range(8):
                for h in range(12):
                    for w in range(12):
                        y[n, c, h, w] = _y[n, c, h, w]

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ft.Array(w_torch.numpy(), device)
    y_shape_torch = torch.zeros(4, dtype=torch.int32)
    y_shape_arr = ft.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(2, 8, 12, 12, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, w_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.nn.functional.conv2d(x_torch, w_torch)
    assert np.array_equal(y_shape_np, [2, 8, 12, 12])
    assert torch.all(torch.isclose(y_torch, y_std))
