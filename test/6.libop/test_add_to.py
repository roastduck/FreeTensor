import torch
import numpy as np

import ir
import ir.libop
from ir.libop import StaticType as T


def test_same_static_shape():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (4, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "output", "cpu")
        "nid: add_to"
        ir.libop.add_to(T("float32", 2), T("float32", 2), "cpu")([4, 4], [4, 4],
                                                                 y, x)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch_new = torch.Tensor(y_arr.numpy().reshape(4, 4))

    assert torch.all(torch.isclose(y_torch_new, x_torch + y_torch))


def test_static_broadcast_shorter():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (4,), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "output", "cpu")
        "nid: add_to"
        ir.libop.add_to(T("float32", 2), T("float32", 1), "cpu")([4, 4], [4], y,
                                                                 x)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(4, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch_new = torch.Tensor(y_arr.numpy().reshape(4, 4))

    assert torch.all(torch.isclose(y_torch_new, x_torch + y_torch))


def test_static_broadcast_1_at_front():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (1, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "output", "cpu")
        "nid: add_to"
        ir.libop.add_to(T("float32", 2), T("float32", 2), "cpu")([4, 4], [1, 4],
                                                                 y, x)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(1, 4, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch_new = torch.Tensor(y_arr.numpy().reshape(4, 4))

    assert torch.all(torch.isclose(y_torch_new, x_torch + y_torch))


def test_static_broadcast_1_at_back():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (4, 1), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "output", "cpu")
        "nid: add_to"
        ir.libop.add_to(T("float32", 2), T("float32", 2), "cpu")([4, 4], [4, 1],
                                                                 y, x)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(4, 1, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch_new = torch.Tensor(y_arr.numpy().reshape(4, 4))

    assert torch.all(torch.isclose(y_torch_new, x_torch + y_torch))


def test_different_dtype():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (4, 4), "int32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "output", "cpu")
        "nid: add_to"
        ir.libop.add_to(T("float32", 2), T("int32", 2), "cpu")([4, 4], [4, 4],
                                                               y, x)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.randint(0, 100, (4, 4), dtype=torch.int32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch_new = torch.Tensor(y_arr.numpy().reshape(4, 4))

    assert torch.all(torch.isclose(y_torch_new, x_torch + y_torch))
