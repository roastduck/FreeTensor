import torch
import numpy as np

import ir
import ir.libop


def test_same_static_shape():
    device = ir.Device(ir.CPU())

    # TODO: Make Python frontend reentrant
    add = ir.libop.add(2, 2, 2, "cpu", "float32")

    @ir.transform
    def f(x, y, out):
        ir.declare_var(x, (4, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "input", "cpu")
        ir.declare_var(out, (4, 4), "float32", "output", "cpu")
        "nid: out_shape"
        ir.declare_var(out_shape, (2,), "int32", "cache", "cpu")
        "nid: add"
        add([4, 4], [4, 4], out_shape, x, y, out)

    print(f)
    s = ir.Schedule(f)
    s.inline("add:V_a_shape")
    s.inline("add:V_b_shape")
    s.inline("out_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ir.Array(out_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy().reshape(4, 4))

    assert torch.all(torch.isclose(out_torch, x_torch + y_torch))


def test_static_broadcast_shorter():
    device = ir.Device(ir.CPU())

    # TODO: Make Python frontend reentrant
    add = ir.libop.add(1, 2, 2, "cpu", "float32")

    @ir.transform
    def f(x, y, out):
        ir.declare_var(x, (4,), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "input", "cpu")
        ir.declare_var(out, (4, 4), "float32", "output", "cpu")
        "nid: out_shape"
        ir.declare_var(out_shape, (2,), "int32", "cache", "cpu")
        "nid: add"
        add([4], [4, 4], out_shape, x, y, out)

    print(f)
    s = ir.Schedule(f)
    s.inline("add:V_a_shape")
    s.inline("add:V_b_shape")
    s.inline("out_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(4, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ir.Array(out_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy().reshape(4, 4))

    assert torch.all(torch.isclose(out_torch, x_torch + y_torch))


def test_static_broadcast_1_at_front():
    device = ir.Device(ir.CPU())

    # TODO: Make Python frontend reentrant
    add = ir.libop.add(2, 2, 2, "cpu", "float32")

    @ir.transform
    def f(x, y, out):
        ir.declare_var(x, (1, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "input", "cpu")
        ir.declare_var(out, (4, 4), "float32", "output", "cpu")
        "nid: out_shape"
        ir.declare_var(out_shape, (2,), "int32", "cache", "cpu")
        "nid: add"
        add([1, 4], [4, 4], out_shape, x, y, out)

    print(f)
    s = ir.Schedule(f)
    s.inline("add:V_a_shape")
    s.inline("add:V_b_shape")
    s.inline("out_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(1, 4, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ir.Array(out_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy().reshape(4, 4))

    assert torch.all(torch.isclose(out_torch, x_torch + y_torch))


def test_static_broadcast_1_at_back():
    device = ir.Device(ir.CPU())

    # TODO: Make Python frontend reentrant
    add = ir.libop.add(2, 2, 2, "cpu", "float32")

    @ir.transform
    def f(x, y, out):
        ir.declare_var(x, (4, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 1), "float32", "input", "cpu")
        ir.declare_var(out, (4, 4), "float32", "output", "cpu")
        "nid: out_shape"
        ir.declare_var(out_shape, (2,), "int32", "cache", "cpu")
        "nid: add"
        add([4, 4], [4, 1], out_shape, x, y, out)

    print(f)
    s = ir.Schedule(f)
    s.inline("add:V_a_shape")
    s.inline("add:V_b_shape")
    s.inline("out_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 1, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ir.Array(out_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy().reshape(4, 4))

    assert torch.all(torch.isclose(out_torch, x_torch + y_torch))
