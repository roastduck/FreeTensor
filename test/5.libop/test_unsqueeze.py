import torch
import numpy as np

import ir
import ir.libop
from ir.libop import StaticType as T


def test_static():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (3, 4, 5), "float32", "input", "cpu")
        ir.declare_var(y, (3, 1, 4, 1, 5), "float32", "output", "cpu")
        "nid: unsqueeze"
        ir.libop.unsqueeze_(T("float32", 3),
                            T("float32", 5),
                            "cpu",
                            axes=[1, 3])([3, 4, 5], [3, 1, 4, 1, 5], x, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("unsqueeze:V_x_shape")
    s.inline("unsqueeze:V_y_shape")
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(3, 1, 4, 1, 5, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(3, 1, 4, 1, 5))

    assert torch.all(torch.isclose(y_torch, x_torch.reshape(3, 1, 4, 1, 5)))


def test_out_of_place():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y_shape, y):
        ir.declare_var(x, (3, 4, 5), "float32", "input", "cpu")
        ir.declare_var(y_shape, (5,), "int32", "output", "cpu")
        ir.declare_var(y, (3, 1, 4, 1, 5), "float32", "output", "cpu")
        "nid: unsqueeze"
        _y = ir.libop.unsqueeze(T("float32", 3),
                                T("float32", 5),
                                "cpu",
                                axes=[1, 3])([3, 4, 5], x)
        for i in range(5):
            y_shape[i] = _y.shape[i]
        for i in range(3):
            for j in range(4):
                for k in range(5):
                    y[i, 0, j, 0, k] = _y[i, 0, j, 0, k]

    print(f)
    s = ir.Schedule(f)
    s.inline("unsqueeze:V_x_shape")
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_shape_torch = torch.zeros(5, dtype=torch.int32)
    y_shape_arr = ir.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(3, 1, 4, 1, 5, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy().reshape(3, 1, 4, 1, 5))

    assert np.array_equal(y_shape_np, [3, 1, 4, 1, 5])
    assert torch.all(torch.isclose(y_torch, x_torch.reshape(3, 1, 4, 1, 5)))
