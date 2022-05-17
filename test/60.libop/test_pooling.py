import torch
import numpy as np

import freetensor as ft
from freetensor import libop


def test_max_pooling_basic():
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(2, 3, 14, 14), "float32", "input", "cpu"]
        y: ft.Var[(2, 3, 12, 12), "float32", "output", "cpu"]
        #! nid: max_pool
        libop.max_pool_(x, y, auto_pad='VALID', kernel_shape=[3, 3])

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 12, 12, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    f(x_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[1, 1])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_same_padding():
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(2, 3, 14, 14), "float32", "input", "cpu"]
        y: ft.Var[(2, 3, 14, 14), "float32", "output", "cpu"]
        #! nid: y_shape
        y_shape = ft.empty((4,), "int32", "cpu")
        #! nid: max_pool
        libop.max_pool_(x, y, auto_pad='SAME_UPPER', kernel_shape=[3, 3])

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 14, 14, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    f(x_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           padding=[1, 1],
                                           kernel_size=[3, 3],
                                           stride=[1, 1])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_stride():
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(2, 3, 12, 12), "float32", "input", "cpu"]
        y: ft.Var[(2, 3, 4, 4), "float32", "output", "cpu"]
        #! nid: max_pool
        libop.max_pool_(x,
                        y,
                        auto_pad='VALID',
                        kernel_shape=[3, 3],
                        strides=[3, 3])

    x_torch = torch.rand(2, 3, 12, 12, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    f(x_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[3, 3])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_dilation():
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(2, 3, 14, 14), "float32", "input", "cpu"]
        y: ft.Var[(2, 3, 10, 10), "float32", "output", "cpu"]
        #! nid: max_pool
        libop.max_pool_(x,
                        y,
                        auto_pad='VALID',
                        kernel_shape=[3, 3],
                        dilations=[2, 2])

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 10, 10, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    f(x_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[1, 1],
                                           dilation=[2, 2])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_out_of_place():
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x):
        x: ft.Var[(2, 3, 14, 14), "float32", "input", "cpu"]
        #! nid: max_pool
        return libop.max_pool(x, auto_pad='VALID', kernel_shape=[3, 3])

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_arr = f(x_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[1, 1])
    assert np.array_equal(y_arr.shape, [2, 3, 12, 12])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_global_avg_pool():
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(2, 3, 14, 14), "float32", "input", "cpu"]
        y: ft.Var[(2, 3), "float32", "output", "cpu"]
        #! nid: max_pool
        libop.global_avg_pool_(x, y)

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    f(x_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.nn.functional.avg_pool2d(x_torch,
                                           kernel_size=[14, 14]).reshape(2, 3)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_global_avg_pool_out_of_place():
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x):
        x: ft.Var[(2, 3, 14, 14), "float32", "input", "cpu"]
        #! nid: max_pool
        return libop.global_avg_pool(x)

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_arr = f(x_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.nn.functional.avg_pool2d(x_torch,
                                           kernel_size=[14, 14]).reshape(2, 3)
    assert np.array_equal(y_arr.shape, [2, 3])
    assert torch.all(torch.isclose(y_torch, y_std))
