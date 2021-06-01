import torch
import numpy as np

import ir
import ir.libop


def test_static_shape():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (4, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((2,), "int32", "cache", "cpu")
        "nid: add"
        ir.libop.relu(2, "cpu", "float32")([4, 4], y_shape, x, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("add:V_x_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(4, 4, dtype=torch.float32) - 0.5
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(4, 4))

    assert torch.all(torch.isclose(y_torch, torch.relu(x_torch)))
