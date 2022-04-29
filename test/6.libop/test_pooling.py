import torch
import numpy as np

import freetensor as ft
from freetensor import libop


def test_max_pooling_basic():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        ft.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ft.declare_var(y, (2, 3, 12, 12), "float32", "output", "cpu")
        "nid: max_pool"
        libop.max_pool_(x, y, auto_pad='VALID', kernel_shape=[3, 3])

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 12, 12, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[1, 1])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_same_padding():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        ft.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ft.declare_var(y, (2, 3, 14, 14), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ft.create_var((4,), "int32", "cpu")
        "nid: max_pool"
        libop.max_pool_(x, y, auto_pad='SAME_UPPER', kernel_shape=[3, 3])

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 14, 14, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           padding=[1, 1],
                                           kernel_size=[3, 3],
                                           stride=[1, 1])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_stride():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        ft.declare_var(x, (2, 3, 12, 12), "float32", "input", "cpu")
        ft.declare_var(y, (2, 3, 4, 4), "float32", "output", "cpu")
        "nid: max_pool"
        libop.max_pool_(x,
                        y,
                        auto_pad='VALID',
                        kernel_shape=[3, 3],
                        strides=[3, 3])

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(2, 3, 12, 12, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[3, 3])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_dilation():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        ft.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ft.declare_var(y, (2, 3, 10, 10), "float32", "output", "cpu")
        "nid: max_pool"
        libop.max_pool_(x,
                        y,
                        auto_pad='VALID',
                        kernel_shape=[3, 3],
                        dilations=[2, 2])

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 10, 10, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[1, 1],
                                           dilation=[2, 2])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_out_of_place():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y_shape, y):
        ft.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ft.declare_var(y_shape, (4,), "int32", "output", "cpu")
        ft.declare_var(y, (2, 3, 12, 12), "float32", "output", "cpu")
        "nid: max_pool"
        _y = libop.max_pool(x, auto_pad='VALID', kernel_shape=[3, 3])
        for i in range(4):
            y_shape[i] = _y.shape(i)
        for n in range(2):
            for c in range(3):
                for h in range(12):
                    for w in range(12):
                        y[n, c, h, w] = _y[n, c, h, w]

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_shape_torch = torch.zeros(4, dtype=torch.int32)
    y_shape_arr = ft.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 12, 12, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[1, 1])
    assert np.array_equal(y_shape_np, [2, 3, 12, 12])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_global_avg_pool():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        ft.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ft.declare_var(y, (2, 3), "float32", "output", "cpu")
        "nid: max_pool"
        libop.global_avg_pool_(x, y)

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.nn.functional.avg_pool2d(x_torch,
                                           kernel_size=[14, 14]).reshape(2, 3)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_global_avg_pool_out_of_place():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y_shape, y):
        ft.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ft.declare_var(y_shape, (2,), "int32", "output", "cpu")
        ft.declare_var(y, (2, 3), "float32", "output", "cpu")
        "nid: max_pool"
        _y = libop.global_avg_pool(x)
        for i in range(2):
            y_shape[i] = _y.shape(i)
        for i in range(2):
            for j in range(3):
                y[i, j] = _y[i, j]

    print(f)
    f = ft.lower(f, ft.CPU())
    print(f)

    code = ft.codegen(f, ft.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_shape_torch = torch.zeros(2, dtype=torch.int32)
    y_shape_arr = ft.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    ft.Driver(f, code, device)(x_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.nn.functional.avg_pool2d(x_torch,
                                           kernel_size=[14, 14]).reshape(2, 3)
    assert np.array_equal(y_shape_np, [2, 3])
    assert torch.all(torch.isclose(y_torch, y_std))
