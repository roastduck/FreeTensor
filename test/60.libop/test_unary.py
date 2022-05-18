import torch
import pytest
import operator
import numpy as np

import freetensor as ft


@pytest.mark.parametrize('libop_func, torch_func, require_positive', [
    (ft.abs_, torch.abs, False),
    (ft.exp_, torch.exp, False),
    (ft.sigmoid_, torch.sigmoid, False),
    (ft.sqrt_, torch.sqrt, True),
    (ft.square_, torch.square, False),
    (ft.relu_, torch.relu, False),
    (ft.tanh_, torch.tanh, False),
    (ft.floor_, torch.floor, False),
    (ft.ceil_, torch.ceil, False),
])
def test_static_shape(libop_func, torch_func, require_positive):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        #! nid: to_test
        libop_func(x, y)

    if require_positive:
        x_torch = torch.rand(4, 4, dtype=torch.float32) * 10
    else:
        x_torch = torch.rand(4, 4, dtype=torch.float32) * 10 - 5
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy(), device)
    f(x_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch, torch_func(x_torch)))


@pytest.mark.parametrize('libop_func, torch_func, require_positive', [
    (ft.abs, torch.abs, False),
    (ft.exp, torch.exp, False),
    (ft.sigmoid, torch.sigmoid, False),
    (ft.sqrt, torch.sqrt, True),
    (ft.square, torch.square, False),
    (ft.relu, torch.relu, False),
    (ft.tanh, torch.tanh, False),
    (ft.floor, torch.floor, False),
    (ft.ceil, torch.ceil, False),
    (ft.neg, operator.neg, False),
    (operator.neg, operator.neg, False),
])
def test_out_of_place(libop_func, torch_func, require_positive):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        #! nid: to_test
        return libop_func(x)

    if require_positive:
        x_torch = torch.rand(4, 4, dtype=torch.float32) * 10
    else:
        x_torch = torch.rand(4, 4, dtype=torch.float32) * 10 - 5
    x_arr = ft.Array(x_torch.numpy(), device)
    y_arr = f(x_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert np.array_equal(y_arr.shape, [4, 4])
    assert torch.all(torch.isclose(y_torch, torch_func(x_torch)))


@pytest.mark.parametrize('libop_func, torch_func, require_positive', [
    (ft.abs_, torch.abs, False),
    (ft.exp_, torch.exp, False),
    (ft.sigmoid_, torch.sigmoid, False),
    (ft.sqrt_, torch.sqrt, True),
    (ft.square_, torch.square, False),
    (ft.relu_, torch.relu, False),
    (ft.tanh_, torch.tanh, False),
    (ft.floor_, torch.floor, False),
    (ft.ceil_, torch.ceil, False),
])
def test_grad(libop_func, torch_func, require_positive):
    device = ft.Device(ft.CPU())

    @ft.transform
    def f(x, y):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        #! nid: to_test
        libop_func(x, y)

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

    if require_positive:
        x_torch = torch.rand(4, 4, dtype=torch.float32) * 10
    else:
        x_torch = torch.rand(4, 4, dtype=torch.float32) * 10 - 5
    x_arr = ft.Array(x_torch.numpy(), device)
    x_torch.requires_grad = True
    y_torch_ours = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch_ours.numpy(), device)
    ft.Driver(f, f_code, device)(x_arr, y_arr)
    y_torch_ours = torch.tensor(y_arr.numpy())
    y_torch = torch_func(x_torch)
    assert torch.all(torch.isclose(y_torch_ours, y_torch))

    y_torch.grad = torch.rand(4, 4, dtype=torch.float32)
    d_y_arr = ft.Array(y_torch.grad.numpy(), device)
    x_grad_torch_ours = torch.zeros(4, 4, dtype=torch.float32)
    d_x_arr = ft.Array(x_grad_torch_ours.numpy(), device)
    kvs = {}
    kvs[privdes['y']] = d_y_arr
    kvs[requires['x']] = d_x_arr
    ft.Driver(g, g_code, device)(**kvs)
    x_grad_torch_ours = torch.tensor(d_x_arr.numpy())
    y_torch.backward(y_torch.grad)
    assert torch.all(torch.isclose(x_grad_torch_ours, x_torch.grad, 1e-4, 1e-7))
