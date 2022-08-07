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
    device = ft.Device(ft.TargetType.CPU)

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
    x_arr = ft.Array(x_torch.numpy())
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy())
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
    device = ft.Device(ft.TargetType.CPU)

    @ft.optimize(device=device, verbose=1)
    def f(x):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        #! nid: to_test
        return libop_func(x)

    if require_positive:
        x_torch = torch.rand(4, 4, dtype=torch.float32) * 10
    else:
        x_torch = torch.rand(4, 4, dtype=torch.float32) * 10 - 5
    x_arr = ft.Array(x_torch.numpy())
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
def test_inplace_grad_of_inplace_func(libop_func, torch_func, require_positive):
    device = ft.Device(ft.TargetType.CPU)

    @ft.transform
    def f(x, y):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        #! nid: to_test
        libop_func(x, y)

    f, g, requires, provides = ft.grad_(f, ["x"], ["y"],
                                        ft.GradTapeMode.NoReuseOnly)
    print("Forward:")
    f = ft.optimize(f, verbose=1)
    print("Backward:")
    g = ft.optimize(g, verbose=1)

    if require_positive:
        x_torch = torch.rand(4, 4, dtype=torch.float32) * 10
    else:
        x_torch = torch.rand(4, 4, dtype=torch.float32) * 10 - 5
    x_arr = ft.Array(x_torch.numpy())
    x_torch.requires_grad = True
    y_torch_ours = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ft.Array(y_torch_ours.numpy())
    f(x_arr, y_arr)
    y_torch_ours = torch.tensor(y_arr.numpy())
    y_torch = torch_func(x_torch)
    assert torch.all(torch.isclose(y_torch_ours, y_torch))

    y_torch.grad = torch.rand(4, 4, dtype=torch.float32)
    d_y_arr = ft.Array(y_torch.grad.numpy())
    x_grad_torch_ours = torch.zeros(4, 4, dtype=torch.float32)
    d_x_arr = ft.Array(x_grad_torch_ours.numpy())
    g(**{provides['y']: d_y_arr, requires['x']: d_x_arr})
    x_grad_torch_ours = torch.tensor(d_x_arr.numpy())
    y_torch.backward(y_torch.grad)
    assert torch.all(torch.isclose(x_grad_torch_ours, x_torch.grad, 1e-4, 1e-7))


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
])
def test_inplace_grad_of_out_of_place_func(libop_func, torch_func,
                                           require_positive):
    device = ft.Device(ft.TargetType.CPU)

    @ft.transform
    def f(x):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        #! nid: to_test
        return libop_func(x)

    f, g, requires, provides = ft.grad_(f, ["x"], [ft.Return()],
                                        ft.GradTapeMode.NoReuseOnly)
    print("Forward:")
    f = ft.optimize(f, verbose=1)
    print("Backward:")
    g = ft.optimize(g, verbose=1)

    if require_positive:
        x_torch = torch.rand(4, 4, dtype=torch.float32) * 10
    else:
        x_torch = torch.rand(4, 4, dtype=torch.float32) * 10 - 5
    x_arr = ft.Array(x_torch.numpy())
    x_torch.requires_grad = True
    y_arr = f(x_arr)
    y_torch_ours = torch.tensor(y_arr.numpy())
    y_torch = torch_func(x_torch)
    assert torch.all(torch.isclose(y_torch_ours, y_torch))

    y_torch.grad = torch.rand(4, 4, dtype=torch.float32)
    d_y_arr = ft.Array(y_torch.grad.numpy())
    x_grad_torch_ours = torch.zeros(4, 4, dtype=torch.float32)
    d_x_arr = ft.Array(x_grad_torch_ours.numpy())
    g(**{provides['y']: d_y_arr, requires['x']: d_x_arr})
    x_grad_torch_ours = torch.tensor(d_x_arr.numpy())
    y_torch.backward(y_torch.grad)
    assert torch.all(torch.isclose(x_grad_torch_ours, x_torch.grad, 1e-4, 1e-7))


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
])
def test_out_of_place_grad_of_out_of_place_func(libop_func, torch_func,
                                                require_positive):
    device = ft.Device(ft.TargetType.CPU)

    @ft.transform
    def f(x):
        x: ft.Var[(4, 4), "float32", "input", "cpu"]
        #! nid: to_test
        return libop_func(x)

    f, g, requires, provides = ft.grad(f, ["x"], [ft.Return()],
                                       ft.GradTapeMode.NoReuseOnly)
    print("Forward:")
    f = ft.optimize(f, verbose=1)
    print("Backward:")
    g = ft.optimize(g, verbose=1)

    if require_positive:
        x_torch = torch.rand(4, 4, dtype=torch.float32) * 10
    else:
        x_torch = torch.rand(4, 4, dtype=torch.float32) * 10 - 5
    x_arr = ft.Array(x_torch.numpy())
    x_torch.requires_grad = True
    y_arr = f(x_arr)
    y_torch_ours = torch.tensor(y_arr.numpy())
    y_torch = torch_func(x_torch)
    assert torch.all(torch.isclose(y_torch_ours, y_torch))

    y_torch.grad = torch.rand(4, 4, dtype=torch.float32)
    d_y_arr = ft.Array(y_torch.grad.numpy())
    d_x_arr = g(**{provides[ft.Return()]: d_y_arr})
    x_grad_torch_ours = torch.tensor(d_x_arr.numpy())
    y_torch.backward(y_torch.grad)
    assert torch.all(torch.isclose(x_grad_torch_ours, x_torch.grad, 1e-4, 1e-7))
