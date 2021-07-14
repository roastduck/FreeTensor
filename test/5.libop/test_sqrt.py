import torch
import numpy as np

import ir
import ir.libop
from ir.libop import StaticType as T


def test_static_shape():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (4, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "output", "cpu")
        "nid: sqrt"
        ir.libop.sqrt_(T("float32", 2), T("float32", 2), "cpu")([4, 4], [4, 4],
                                                                x, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("sqrt:V_x_shape")
    s.inline("sqrt:V_y_shape")
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(4, 4))

    assert torch.all(torch.isclose(y_torch, torch.sqrt(x_torch)))


def test_out_of_place():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y_shape, y):
        ir.declare_var(x, (4, 4), "float32", "input", "cpu")
        ir.declare_var(y_shape, (2,), "int32", "output", "cpu")
        ir.declare_var(y, (4, 4), "float32", "output", "cpu")
        "nid: sqrt"
        _y = ir.libop.sqrt(T("float32", 2), T("float32", 2), "cpu")([4, 4], x)
        y_shape[0] = _y.shape[0]
        y_shape[1] = _y.shape[1]
        for i in range(4):
            for j in range(4):
                y[i, j] = _y[i, j]

    print(f)
    s = ir.Schedule(f)
    s.inline("sqrt:V_x_shape")
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_shape_torch = torch.zeros(2, dtype=torch.int32)
    y_shape_arr = ir.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy().reshape(4, 4))

    assert np.array_equal(y_shape_np, [4, 4])
    assert torch.all(torch.isclose(y_torch, torch.sqrt(x_torch)))
