import torch
import numpy as np

import freetensor as ft
from freetensor import libop


def test_static_shape():
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: softmax"
        libop.softmax_(x, y)

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    f(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch, torch.softmax(x_torch, axis=-1)))


def test_out_of_place():
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y_shape, y):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y_shape: ft.Var[(2,), "int32", "output", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: softmax"
        _y = libop.softmax(x, axis=-1)
        y_shape[0] = _y.shape(0)
        y_shape[1] = _y.shape(1)
        for i in range(4):
            for j in range(4):
                y[i, j] = _y[i, j]

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_shape_torch = torch.zeros(2, dtype=torch.int32)
    y_shape_arr = ft.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    f(x_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy())

    assert np.array_equal(y_shape_np, [4, 4])
    assert torch.all(torch.isclose(y_torch, torch.softmax(x_torch, axis=-1)))


def test_grad():
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        "nid: softmax"
        libop.softmax_(x, y)

    print(f)
    f, g, requires, privdes, _ = ft.grad(f, set(["x"]), set(["y"]),
                                         ft.GradTapeMode.NoReuseOnly)
    print("Forward:")
    print(f)
    print("Backward:")
    print(g)
    f = ft.lower(f, ft.CPU())
    print("Forward:")
    print(f)
    g = ft.lower(g, ft.CPU())
    print("Backward:")
    print(g)

    f_code = ft.codegen(f, ft.CPU())
    g_code = ft.codegen(g, ft.CPU())

    def get_shape_and_dtype(func, nid):
        s = ft.Schedule(func)
        vardef = s.find(nid).node()
        shape = []
        for x in vardef.buffer.tensor.shape:
            assert isinstance(x, ft.ffi.IntConst)
            shape.append(x.val)
        if vardef.buffer.tensor.dtype == ft.DataType.Float32:
            dtype = torch.float32
        elif vardef.buffer.tensor.dtype == ft.DataType.Int32:
            dtype = torch.int32
        else:
            assert False
        return shape, dtype

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy(), device)
    x_torch.requires_grad = True
    y_torch_ours = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch_ours.numpy(), device)
    ft.Driver(f, f_code, device)(x_arr, y_arr)
    y_torch_ours = torch.Tensor(y_arr.numpy())
    y_torch = torch.softmax(x_torch, axis=-1)
    assert torch.all(torch.isclose(y_torch_ours, y_torch))

    y_torch.grad = torch.rand(4, 4, dtype=torch.float32)
    d_y_arr = ft.Array(y_torch.grad.numpy(), device)
    x_grad_torch_ours = torch.zeros(4, 4, dtype=torch.float32)
    d_x_arr = ft.Array(x_grad_torch_ours.numpy(), device)
    kvs = {}
    kvs[privdes['y']] = d_y_arr
    kvs[requires['x']] = d_x_arr
    ft.Driver(g, g_code, device)(x_arr, y_arr, **kvs)
    x_grad_torch_ours = torch.Tensor(d_x_arr.numpy())
    y_torch.backward(y_torch.grad)
    assert torch.all(torch.isclose(x_grad_torch_ours, x_torch.grad, 1e-4, 1e-7))
