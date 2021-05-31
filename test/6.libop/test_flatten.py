import torch
import numpy as np

import ir
import ir.libop


def test_static():
    device = ir.Device(ir.CPU())

    # TODO: Make Python frontend reentrant
    flatten = ir.libop.flatten(3, "cpu", "float32")

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (3, 4, 5), "float32", "input", "cpu")
        ir.declare_var(y, (3, 20), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((2,), "int32", "cache", "cpu")
        "nid: flatten"
        flatten([3, 4, 5], y_shape, x, y)

    print(f)
    s = ir.Schedule(f)
    s.unroll("flatten:L_shape", True)
    s.inline("flatten:V_x_shape")
    s.inline("flatten:recur:V_recur_y_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(3, 20, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(3, 20))

    assert torch.all(torch.isclose(y_torch, x_torch.reshape(3, -1)))
