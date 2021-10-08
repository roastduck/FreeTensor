import torch
import numpy as np

import ir
import ir.libop
from ir.libop import StaticType as T


def test_same_static_shape():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y, out):
        ir.declare_var(x, (4, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "input", "cpu")
        ir.declare_var(out, (4, 4), "float32", "output", "cpu")
        "nid: add"
        ir.libop.add_(T("float32", 2), T("float32", 2), T("float32", 2),
                      "cpu")(ir.Tensor([4, 4], "cpu"), ir.Tensor([4, 4], "cpu"),
                             ir.Tensor([4, 4], "cpu"), x, y, out)

    print(f)
    s = ir.Schedule(f)
    s.inline("add:a_shape")
    s.inline("add:b_shape")
    s.inline("add:out_shape")
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

    @ir.transform
    def f(x, y, out):
        ir.declare_var(x, (4,), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "input", "cpu")
        ir.declare_var(out, (4, 4), "float32", "output", "cpu")
        "nid: add"
        ir.libop.add_(T("float32", 1), T("float32", 2), T("float32", 2),
                      "cpu")(ir.Tensor([4], "cpu"), ir.Tensor([4, 4], "cpu"),
                             ir.Tensor([4, 4], "cpu"), x, y, out)

    print(f)
    s = ir.Schedule(f)
    s.inline("add:a_shape")
    s.inline("add:b_shape")
    s.inline("add:out_shape")
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

    @ir.transform
    def f(x, y, out):
        ir.declare_var(x, (1, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "input", "cpu")
        ir.declare_var(out, (4, 4), "float32", "output", "cpu")
        "nid: out_shape"
        out_shape = ir.create_var((2,), "int32", "cache", "cpu")
        "nid: add"
        ir.libop.add_(T("float32", 2), T("float32", 2), T("float32", 2),
                      "cpu")(ir.Tensor([1, 4], "cpu"), ir.Tensor([4, 4], "cpu"),
                             ir.Tensor([4, 4], "cpu"), x, y, out)

    print(f)
    s = ir.Schedule(f)
    s.inline("add:a_shape")
    s.inline("add:b_shape")
    s.inline("add:out_shape")
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

    @ir.transform
    def f(x, y, out):
        ir.declare_var(x, (4, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 1), "float32", "input", "cpu")
        ir.declare_var(out, (4, 4), "float32", "output", "cpu")
        "nid: out_shape"
        out_shape = ir.create_var((2,), "int32", "cache", "cpu")
        "nid: add"
        ir.libop.add_(T("float32", 2), T("float32", 2), T("float32", 2),
                      "cpu")(ir.Tensor([4, 4], "cpu"), ir.Tensor([4, 1], "cpu"),
                             ir.Tensor([4, 4], "cpu"), x, y, out)

    print(f)
    s = ir.Schedule(f)
    s.inline("add:a_shape")
    s.inline("add:b_shape")
    s.inline("add:out_shape")
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


def test_different_dtype():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y, out):
        ir.declare_var(x, (4, 4), "int32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "input", "cpu")
        ir.declare_var(out, (4, 4), "float32", "output", "cpu")
        "nid: out_shape"
        out_shape = ir.create_var((2,), "int32", "cache", "cpu")
        "nid: add"
        ir.libop.add_(T("int32", 2), T("float32", 2), T("float32", 2),
                      "cpu")(ir.Tensor([4, 4], "cpu"), ir.Tensor([4, 4], "cpu"),
                             ir.Tensor([4, 4], "cpu"), x, y, out)

    print(f)
    s = ir.Schedule(f)
    s.inline("add:a_shape")
    s.inline("add:b_shape")
    s.inline("add:out_shape")
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.randint(0, 100, (4, 4), dtype=torch.int32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ir.Array(out_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr, out_arr)
    out_torch = torch.Tensor(out_arr.numpy().reshape(4, 4))

    assert torch.all(torch.isclose(out_torch, x_torch + y_torch))


def test_out_of_place():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y, out_shape, out):
        ir.declare_var(x, (4, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "input", "cpu")
        ir.declare_var(out_shape, (2,), "int32", "output", "cpu")
        ir.declare_var(out, (4, 4), "float32", "output", "cpu")
        "nid: add"
        _out = ir.libop.add(T("float32", 2), T("float32", 2), T("float32", 2),
                            "cpu")(ir.Tensor([4, 4], "cpu"),
                                   ir.Tensor([4, 4], "cpu"), x, y)
        for i in range(2):
            out_shape[i] = _out.shape[i]
        for i in range(4):
            for j in range(4):
                out[i, j] = _out[i, j]

    print(f)
    s = ir.Schedule(f)
    s.inline("add:a_shape")
    s.inline("add:b_shape")
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.rand(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    out_shape_torch = torch.zeros(2, dtype=torch.int32)
    out_shape_arr = ir.Array(out_shape_torch.numpy(), device)
    out_torch = torch.zeros(4, 4, dtype=torch.float32)
    out_arr = ir.Array(out_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr, out_shape_arr, out_arr)
    out_shape_numpy = out_shape_arr.numpy()
    out_torch = torch.Tensor(out_arr.numpy().reshape(4, 4))

    assert np.array_equal(out_shape_numpy, [4, 4])
    assert torch.all(torch.isclose(out_torch, x_torch + y_torch))
