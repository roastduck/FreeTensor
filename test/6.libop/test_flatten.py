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
        ir.declare_var(y, (3, 20), "float32", "output", "cpu")
        "nid: flatten"
        ir.libop.flatten_(T("float32", 3), T("float32", 2),
                          "cpu")(ir.Tensor([3, 4, 5], "cpu"),
                                 ir.Tensor([3, 20], "cpu"), x, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("flatten:x_shape")
    s.inline("flatten:y_shape")
    s.inline("flatten:recur_y_shape")
    s.inline("flatten:recur:recur:recur_y_shape")
    s.inline("flatten:recur:recur:recur:recur_y_shape")
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


def test_axis():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (3, 4, 5), "float32", "input", "cpu")
        ir.declare_var(y, (12, 5), "float32", "output", "cpu")
        "nid: flatten"
        ir.libop.flatten_(T("float32", 3), T("float32", 2), "cpu",
                          axis=2)(ir.Tensor([3, 4, 5], "cpu"),
                                  ir.Tensor([12, 5], "cpu"), x, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("flatten:x_shape")
    s.inline("flatten:y_shape")
    s.inline("flatten:recur_y_shape")
    s.inline("flatten:recur:recur_y_shape")
    s.inline("flatten:recur:recur:recur:recur_y_shape")
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(12, 5, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(12, 5))

    assert torch.all(torch.isclose(y_torch, x_torch.reshape(-1, 5)))


def test_out_of_place():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y_shape, y):
        ir.declare_var(x, (3, 4, 5), "float32", "input", "cpu")
        ir.declare_var(y_shape, (2,), "int32", "output", "cpu")
        ir.declare_var(y, (3, 20), "float32", "output", "cpu")
        "nid: flatten"
        _y = ir.libop.flatten(T("float32", 3), T("float32", 2),
                              "cpu")(ir.Tensor([3, 4, 5], "cpu"), x)
        for i in range(2):
            y_shape[i] = _y.shape[i]
        for i in range(3):
            for j in range(20):
                y[i, j] = _y[i, j]

    print(f)
    s = ir.Schedule(f)
    #s.inline("flatten:V_x_shape")
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(3, 4, 5, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_shape_torch = torch.zeros(2, dtype=torch.int32)
    y_shape_arr = ir.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(3, 20, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy().reshape(3, 20))

    assert np.array_equal(y_shape_np, [3, 20])
    assert torch.all(torch.isclose(y_torch, x_torch.reshape(3, -1)))
