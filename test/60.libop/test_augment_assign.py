import torch
import pytest
import operator
import functools
import numpy as np

import freetensor as ft
from freetensor import libop


def rand(*shape, **kvs):
    dtype = kvs["dtype"]
    if dtype == "float32":
        return torch.rand(*shape, dtype=torch.float32) + 1
    if dtype == "int32":
        return torch.randint(1, 100, shape, dtype=torch.int32)
    assert False


@pytest.mark.parametrize('libop_func, torch_func, dtype', [
    (libop.add_to, operator.add, "float32"),
    (libop.sub_to, operator.sub, "float32"),
    (libop.mul_to, operator.mul, "float32"),
    (libop.truediv_to, operator.truediv, "float32"),
    (libop.floordiv_to, functools.partial(torch.div,
                                          rounding_mode='floor'), "int32"),
    (libop.mod_to, operator.mod, "int32"),
])
def test_same_static_shape(libop_func, torch_func, dtype):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4, 4), dtype, "input", "cpu"]
        y: ft.Var[(4, 4), dtype, "output", "cpu"]
        #! nid: to_test
        libop_func(y, x)

    x_torch = rand(4, 4, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy())
    y_torch = rand(4, 4, dtype=dtype)
    y_arr = ft.Array(y_torch.numpy().copy())
    f(x_arr, y_arr)
    y_torch_new = torch.tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch_new, torch_func(y_torch, x_torch)))


@pytest.mark.parametrize('libop_func, torch_func, dtype', [
    (libop.add_to, operator.add, "float32"),
    (libop.sub_to, operator.sub, "float32"),
    (libop.mul_to, operator.mul, "float32"),
    (libop.truediv_to, operator.truediv, "float32"),
    (libop.floordiv_to, functools.partial(torch.div,
                                          rounding_mode='floor'), "int32"),
    (libop.mod_to, operator.mod, "int32"),
])
def test_static_broadcast_shorter(libop_func, torch_func, dtype):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4,), dtype, "input", "cpu"]
        y: ft.Var[(4, 4), dtype, "output", "cpu"]
        #! nid: to_test
        libop_func(y, x)

    x_torch = rand(4, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy())
    y_torch = rand(4, 4, dtype=dtype)
    y_arr = ft.Array(y_torch.numpy().copy())
    f(x_arr, y_arr)
    y_torch_new = torch.tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch_new, torch_func(y_torch, x_torch)))


@pytest.mark.parametrize('libop_func, torch_func, dtype', [
    (libop.add_to, operator.add, "float32"),
    (libop.sub_to, operator.sub, "float32"),
    (libop.mul_to, operator.mul, "float32"),
    (libop.truediv_to, operator.truediv, "float32"),
    (libop.floordiv_to, functools.partial(torch.div,
                                          rounding_mode='floor'), "int32"),
    (libop.mod_to, operator.mod, "int32"),
])
def test_static_broadcast_1_at_front(libop_func, torch_func, dtype):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(1, 4), dtype, "input", "cpu"]
        y: ft.Var[(4, 4), dtype, "output", "cpu"]
        #! nid: to_test
        libop_func(y, x)

    x_torch = rand(1, 4, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy())
    y_torch = rand(4, 4, dtype=dtype)
    y_arr = ft.Array(y_torch.numpy().copy())
    f(x_arr, y_arr)
    y_torch_new = torch.tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch_new, torch_func(y_torch, x_torch)))


@pytest.mark.parametrize('libop_func, torch_func, dtype', [
    (libop.add_to, operator.add, "float32"),
    (libop.sub_to, operator.sub, "float32"),
    (libop.mul_to, operator.mul, "float32"),
    (libop.truediv_to, operator.truediv, "float32"),
    (libop.floordiv_to, functools.partial(torch.div,
                                          rounding_mode='floor'), "int32"),
    (libop.mod_to, operator.mod, "int32"),
])
def test_static_broadcast_1_at_back(libop_func, torch_func, dtype):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4, 1), dtype, "input", "cpu"]
        y: ft.Var[(4, 4), dtype, "output", "cpu"]
        #! nid: to_test
        libop_func(y, x)

    x_torch = rand(4, 1, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy())
    y_torch = rand(4, 4, dtype=dtype)
    y_arr = ft.Array(y_torch.numpy().copy())
    f(x_arr, y_arr)
    y_torch_new = torch.tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch_new, torch_func(y_torch, x_torch)))


@pytest.mark.parametrize('libop_func, torch_func', [
    (libop.add_to, operator.add),
    (libop.sub_to, operator.sub),
    (libop.mul_to, operator.mul),
    (libop.truediv_to, operator.truediv),
])
def test_different_dtype(libop_func, torch_func):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4, 4), "int32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        #! nid: to_test
        libop_func(y, x)

    x_torch = rand(4, 4, dtype="int32")
    x_arr = ft.Array(x_torch.numpy())
    y_torch = rand(4, 4, dtype="float32")
    y_arr = ft.Array(y_torch.numpy().copy())
    f(x_arr, y_arr)
    y_torch_new = torch.tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch_new, torch_func(y_torch, x_torch)))


@pytest.mark.parametrize('libop_func, torch_func, dtype', [
    (operator.iadd, operator.add, "float32"),
    (operator.isub, operator.sub, "float32"),
    (operator.imul, operator.mul, "float32"),
    (operator.itruediv, operator.truediv, "float32"),
    (operator.ifloordiv, functools.partial(torch.div,
                                           rounding_mode='floor'), "int32"),
    (operator.imod, operator.mod, "int32"),
])
def test_operator_overload(libop_func, torch_func, dtype):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4, 4), dtype, "input", "cpu"]
        y: ft.Var[(4, 4), dtype, "output", "cpu"]
        #! nid: to_test
        libop_func(y[:], x)  # E.g., y[:] += x

    x_torch = rand(4, 4, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy())
    y_torch = rand(4, 4, dtype=dtype)
    y_arr = ft.Array(y_torch.numpy().copy())
    f(x_arr, y_arr)
    y_torch_new = torch.tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch_new, torch_func(y_torch, x_torch)))
