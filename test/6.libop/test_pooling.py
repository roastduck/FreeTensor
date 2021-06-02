import torch
import numpy as np

import ir
import ir.libop
from ir.libop import StaticType as T


def test_max_pooling_basic():
    device = ir.Device(ir.CPU())

    max_pool = ir.libop.max_pool(T("float32", 4),
                                 "cpu",
                                 auto_pad='VALID',
                                 kernel_shape=[3, 3])

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(y, (2, 3, 12, 12), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((4,), "int32", "cache", "cpu")
        "nid: max_pool"
        max_pool([2, 3, 14, 14], y_shape, x, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("max_pool:V_X_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 12, 12, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 3, 12, 12))

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[1, 1])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_same_padding():
    device = ir.Device(ir.CPU())

    max_pool = ir.libop.max_pool(T("float32", 4),
                                 "cpu",
                                 auto_pad='SAME_UPPER',
                                 kernel_shape=[3, 3])

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(y, (2, 3, 14, 14), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((4,), "int32", "cache", "cpu")
        "nid: max_pool"
        max_pool([2, 3, 14, 14], y_shape, x, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("max_pool:V_X_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 14, 14, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 3, 14, 14))

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           padding=[1, 1],
                                           kernel_size=[3, 3],
                                           stride=[1, 1])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_stride():
    device = ir.Device(ir.CPU())

    max_pool = ir.libop.max_pool(T("float32", 4),
                                 "cpu",
                                 auto_pad='VALID',
                                 kernel_shape=[3, 3],
                                 strides=[3, 3])

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (2, 3, 12, 12), "float32", "input", "cpu")
        ir.declare_var(y, (2, 3, 4, 4), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((4,), "int32", "cache", "cpu")
        "nid: max_pool"
        max_pool([2, 3, 14, 14], y_shape, x, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("max_pool:V_X_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 12, 12, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 3, 4, 4))

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[3, 3])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_dilation():
    device = ir.Device(ir.CPU())

    max_pool = ir.libop.max_pool(T("float32", 4),
                                 "cpu",
                                 auto_pad='VALID',
                                 kernel_shape=[3, 3],
                                 dilations=[2, 2])

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(y, (2, 3, 10, 10), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((4,), "int32", "cache", "cpu")
        "nid: max_pool"
        max_pool([2, 3, 14, 14], y_shape, x, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("max_pool:V_X_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 10, 10, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 3, 10, 10))

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[1, 1],
                                           dilation=[2, 2])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_global_avg_pool():
    device = ir.Device(ir.CPU())

    ga_pool = ir.libop.global_avg_pool(T("float32", 4), "cpu")

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(y, (2, 3), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((2,), "int32", "cache", "cpu")
        "nid: max_pool"
        ga_pool([2, 3, 14, 14], y_shape, x, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("max_pool:V_X_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 3))

    y_std = torch.nn.functional.avg_pool2d(x_torch,
                                           kernel_size=[14, 14]).reshape(2, 3)
    assert torch.all(torch.isclose(y_torch, y_std))
