import torch
import numpy as np

import ir
import ir.libop
from ir.libop import StaticType as T


def test_basic():
    device = ir.Device(ir.CPU())

    conv = ir.libop.conv(T("float32", 4),
                         T("float32", 4),
                         None,
                         T("float32", 4),
                         "cpu",
                         auto_pad='VALID')

    @ir.transform
    def f(x, w, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(w, (8, 3, 3, 3), "float32", "input", "cpu")
        ir.declare_var(y, (2, 8, 12, 12), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((4,), "int32", "cache", "cpu")
        "nid: conv"
        conv([2, 3, 14, 14], [8, 3, 3, 3], y_shape, x, w, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("conv:V_X_shape")
    s.inline("conv:V_W_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ir.Array(w_torch.numpy(), device)
    y_torch = torch.zeros(2, 8, 12, 12, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, w_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 8, 12, 12))

    y_std = torch.nn.functional.conv2d(x_torch, w_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias():
    device = ir.Device(ir.CPU())

    conv = ir.libop.conv(T("float32", 4),
                         T("float32", 4),
                         T("float32", 4),
                         T("float32", 4),
                         "cpu",
                         auto_pad='VALID')

    @ir.transform
    def f(x, w, b, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(w, (8, 3, 3, 3), "float32", "input", "cpu")
        ir.declare_var(b, (8,), "float32", "input", "cpu")
        ir.declare_var(y, (2, 8, 12, 12), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((4,), "int32", "cache", "cpu")
        "nid: conv"
        conv([2, 3, 14, 14], [8, 3, 3, 3], [8], y_shape, x, w, b, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("conv:V_X_shape")
    s.inline("conv:V_W_shape")
    s.inline("conv:V_B_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ir.Array(w_torch.numpy(), device)
    b_torch = torch.rand(8, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(2, 8, 12, 12, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, w_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 8, 12, 12))

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, bias=b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_same_pad():
    device = ir.Device(ir.CPU())

    conv = ir.libop.conv(T("float32", 4),
                         T("float32", 4),
                         None,
                         T("float32", 4),
                         "cpu",
                         kernel_shape=(3, 3),
                         auto_pad='SAME_UPPER')

    @ir.transform
    def f(x, w, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(w, (8, 3, 3, 3), "float32", "input", "cpu")
        ir.declare_var(y, (2, 8, 14, 14), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((4,), "int32", "cache", "cpu")
        "nid: conv"
        conv([2, 3, 14, 14], [8, 3, 3, 3], y_shape, x, w, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("conv:V_X_shape")
    s.inline("conv:V_W_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ir.Array(w_torch.numpy(), device)
    y_torch = torch.zeros(2, 8, 14, 14, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, w_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 8, 14, 14))

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, padding=[1, 1])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_stride():
    device = ir.Device(ir.CPU())

    conv = ir.libop.conv(T("float32", 4),
                         T("float32", 4),
                         None,
                         T("float32", 4),
                         "cpu",
                         auto_pad='VALID',
                         strides=(2, 2))

    @ir.transform
    def f(x, w, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(w, (8, 3, 3, 3), "float32", "input", "cpu")
        ir.declare_var(y, (2, 8, 6, 6), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((4,), "int32", "cache", "cpu")
        "nid: conv"
        conv([2, 3, 14, 14], [8, 3, 3, 3], y_shape, x, w, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("conv:V_X_shape")
    s.inline("conv:V_W_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ir.Array(w_torch.numpy(), device)
    y_torch = torch.zeros(2, 8, 6, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, w_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 8, 6, 6))

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, stride=(2, 2))
    assert torch.all(torch.isclose(y_torch, y_std))


def test_group():
    device = ir.Device(ir.CPU())

    conv = ir.libop.conv(T("float32", 4),
                         T("float32", 4),
                         None,
                         T("float32", 4),
                         "cpu",
                         auto_pad='VALID',
                         group=2)

    @ir.transform
    def f(x, w, y):
        ir.declare_var(x, (2, 4, 14, 14), "float32", "input", "cpu")
        ir.declare_var(w, (8, 2, 3, 3), "float32", "input", "cpu")
        ir.declare_var(y, (2, 8, 12, 12), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((4,), "int32", "cache", "cpu")
        "nid: conv"
        conv([2, 4, 14, 14], [8, 2, 3, 3], y_shape, x, w, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("conv:V_X_shape")
    s.inline("conv:V_W_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 4, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    w_torch = torch.rand(8, 2, 3, 3, dtype=torch.float32)
    w_arr = ir.Array(w_torch.numpy(), device)
    y_torch = torch.zeros(2, 8, 12, 12, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, w_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 8, 12, 12))

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, groups=2)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_dilation():
    device = ir.Device(ir.CPU())

    conv = ir.libop.conv(T("float32", 4),
                         T("float32", 4),
                         None,
                         T("float32", 4),
                         "cpu",
                         auto_pad='VALID',
                         dilations=(2, 2))

    @ir.transform
    def f(x, w, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(w, (8, 3, 3, 3), "float32", "input", "cpu")
        ir.declare_var(y, (2, 8, 10, 10), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((4,), "int32", "cache", "cpu")
        "nid: conv"
        conv([2, 3, 14, 14], [8, 3, 3, 3], y_shape, x, w, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("conv:V_X_shape")
    s.inline("conv:V_W_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    w_torch = torch.rand(8, 3, 3, 3, dtype=torch.float32)
    w_arr = ir.Array(w_torch.numpy(), device)
    y_torch = torch.zeros(2, 8, 10, 10, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, w_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 8, 10, 10))

    y_std = torch.nn.functional.conv2d(x_torch, w_torch, dilation=(2, 2))
    assert torch.all(torch.isclose(y_torch, y_std))
