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
    if dtype == "bool":
        return torch.randint(0, 1, shape, dtype=torch.bool)
    assert False


def zeros(*shape, **kvs):
    dtype = kvs["dtype"]
    if dtype == "float32":
        return torch.zeros(*shape, dtype=torch.float32)
    if dtype == "int32":
        return torch.zeros(*shape, dtype=torch.int32)
    if dtype == "bool":
        return torch.zeros(*shape, dtype=torch.bool)
    assert False


def same(lhs, rhs, dtype):
    if dtype == "float32":
        return torch.all(torch.isclose(lhs, rhs))
    if dtype == "int32":
        return torch.all(lhs == rhs)
    if dtype == "bool":
        return torch.all(lhs == rhs)
    assert False


@pytest.mark.parametrize('libop_func, torch_func, dtype, ret_dtype', [
    (libop.add_, operator.add, "float32", "float32"),
    (libop.sub_, operator.sub, "float32", "float32"),
    (libop.mul_, operator.mul, "float32", "float32"),
    (libop.truediv_, operator.truediv, "float32", "float32"),
    (libop.floordiv_, functools.partial(
        torch.div, rounding_mode='floor'), "int32", "int32"),
    (libop.mod_, operator.mod, "int32", "int32"),
    (libop.l_and_, torch.logical_and, "bool", "bool"),
    (libop.l_or_, torch.logical_or, "bool", "bool"),
    (libop.lt_, operator.lt, "int32", "bool"),
    (libop.le_, operator.le, "int32", "bool"),
    (libop.gt_, operator.gt, "int32", "bool"),
    (libop.ge_, operator.ge, "int32", "bool"),
    (libop.eq_, operator.eq, "int32", "bool"),
    (libop.ne_, operator.ne, "int32", "bool"),
])
def test_same_static_shape(libop_func, torch_func, dtype, ret_dtype):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y, out):
        x: ft.Var[(4, 4), dtype, "input", "cpu"]
        y: ft.Var[(4, 4), dtype, "input", "cpu"]
        out: ft.Var[(4, 4), ret_dtype, "output", "cpu"]
        "nid: to_test"
        libop_func(x, y, out)

    x_torch = rand(4, 4, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = rand(4, 4, dtype=dtype)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = zeros(4, 4, dtype=ret_dtype)
    out_arr = ft.Array(out_torch.numpy(), device)
    f(x_arr, y_arr, out_arr)
    out_torch = torch.tensor(out_arr.numpy())

    assert same(out_torch, torch_func(x_torch, y_torch), dtype=ret_dtype)


@pytest.mark.parametrize('libop_func, torch_func, dtype, ret_dtype', [
    (libop.add_, operator.add, "float32", "float32"),
    (libop.sub_, operator.sub, "float32", "float32"),
    (libop.mul_, operator.mul, "float32", "float32"),
    (libop.truediv_, operator.truediv, "float32", "float32"),
    (libop.floordiv_, functools.partial(
        torch.div, rounding_mode='floor'), "int32", "int32"),
    (libop.mod_, operator.mod, "int32", "int32"),
    (libop.l_and_, torch.logical_and, "bool", "bool"),
    (libop.l_or_, torch.logical_or, "bool", "bool"),
    (libop.lt_, operator.lt, "int32", "bool"),
    (libop.le_, operator.le, "int32", "bool"),
    (libop.gt_, operator.gt, "int32", "bool"),
    (libop.ge_, operator.ge, "int32", "bool"),
    (libop.eq_, operator.eq, "int32", "bool"),
    (libop.ne_, operator.ne, "int32", "bool"),
])
def test_static_broadcast_shorter(libop_func, torch_func, dtype, ret_dtype):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y, out):
        x: ft.Var[(4,), dtype, "input", "cpu"]
        y: ft.Var[(4, 4), dtype, "input", "cpu"]
        out: ft.Var[(4, 4), ret_dtype, "output", "cpu"]
        "nid: to_test"
        libop_func(x, y, out)

    x_torch = rand(4, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = rand(4, 4, dtype=dtype)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = zeros(4, 4, dtype=ret_dtype)
    out_arr = ft.Array(out_torch.numpy(), device)
    f(x_arr, y_arr, out_arr)
    out_torch = torch.tensor(out_arr.numpy())

    assert same(out_torch, torch_func(x_torch, y_torch), dtype=ret_dtype)


@pytest.mark.parametrize('libop_func, torch_func, dtype, ret_dtype', [
    (libop.add_, operator.add, "float32", "float32"),
    (libop.sub_, operator.sub, "float32", "float32"),
    (libop.mul_, operator.mul, "float32", "float32"),
    (libop.truediv_, operator.truediv, "float32", "float32"),
    (libop.floordiv_, functools.partial(
        torch.div, rounding_mode='floor'), "int32", "int32"),
    (libop.mod_, operator.mod, "int32", "int32"),
    (libop.l_and_, torch.logical_and, "bool", "bool"),
    (libop.l_or_, torch.logical_or, "bool", "bool"),
    (libop.lt_, operator.lt, "int32", "bool"),
    (libop.le_, operator.le, "int32", "bool"),
    (libop.gt_, operator.gt, "int32", "bool"),
    (libop.ge_, operator.ge, "int32", "bool"),
    (libop.eq_, operator.eq, "int32", "bool"),
    (libop.ne_, operator.ne, "int32", "bool"),
])
def test_static_broadcast_1_at_front(libop_func, torch_func, dtype, ret_dtype):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y, out):
        x: ft.Var[(1, 4), dtype, "input", "cpu"]
        y: ft.Var[(4, 4), dtype, "input", "cpu"]
        out: ft.Var[(4, 4), ret_dtype, "output", "cpu"]
        "nid: to_test"
        libop_func(x, y, out)

    x_torch = rand(1, 4, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = rand(4, 4, dtype=dtype)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = zeros(4, 4, dtype=ret_dtype)
    out_arr = ft.Array(out_torch.numpy(), device)
    f(x_arr, y_arr, out_arr)
    out_torch = torch.tensor(out_arr.numpy())

    assert same(out_torch, torch_func(x_torch, y_torch), dtype=ret_dtype)


@pytest.mark.parametrize('libop_func, torch_func, dtype, ret_dtype', [
    (libop.add_, operator.add, "float32", "float32"),
    (libop.sub_, operator.sub, "float32", "float32"),
    (libop.mul_, operator.mul, "float32", "float32"),
    (libop.truediv_, operator.truediv, "float32", "float32"),
    (libop.floordiv_, functools.partial(
        torch.div, rounding_mode='floor'), "int32", "int32"),
    (libop.mod_, operator.mod, "int32", "int32"),
    (libop.l_and_, torch.logical_and, "bool", "bool"),
    (libop.l_or_, torch.logical_or, "bool", "bool"),
    (libop.lt_, operator.lt, "int32", "bool"),
    (libop.le_, operator.le, "int32", "bool"),
    (libop.gt_, operator.gt, "int32", "bool"),
    (libop.ge_, operator.ge, "int32", "bool"),
    (libop.eq_, operator.eq, "int32", "bool"),
    (libop.ne_, operator.ne, "int32", "bool"),
])
def test_static_broadcast_1_at_back(libop_func, torch_func, dtype, ret_dtype):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y, out):
        x: ft.Var[(4, 4), dtype, "input", "cpu"]
        y: ft.Var[(4, 1), dtype, "input", "cpu"]
        out: ft.Var[(4, 4), ret_dtype, "output", "cpu"]
        "nid: to_test"
        libop_func(x, y, out)

    x_torch = rand(4, 4, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = rand(4, 1, dtype=dtype)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = zeros(4, 4, dtype=ret_dtype)
    out_arr = ft.Array(out_torch.numpy(), device)
    f(x_arr, y_arr, out_arr)
    out_torch = torch.tensor(out_arr.numpy())

    assert same(out_torch, torch_func(x_torch, y_torch), dtype=ret_dtype)


@pytest.mark.parametrize('libop_func, torch_func, dtype1, dtype2, ret_dtype', [
    (libop.add_, operator.add, "float32", "int32", "float32"),
    (libop.sub_, operator.sub, "float32", "int32", "float32"),
    (libop.mul_, operator.mul, "float32", "int32", "float32"),
    (libop.truediv_, operator.truediv, "float32", "int32", "float32"),
    (libop.lt_, operator.lt, "float32", "int32", "bool"),
    (libop.le_, operator.le, "float32", "int32", "bool"),
    (libop.gt_, operator.gt, "float32", "int32", "bool"),
    (libop.ge_, operator.ge, "float32", "int32", "bool"),
    (libop.eq_, operator.eq, "float32", "int32", "bool"),
    (libop.ne_, operator.ne, "float32", "int32", "bool"),
])
def test_different_dtype(libop_func, torch_func, dtype1, dtype2, ret_dtype):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y, out):
        x: ft.Var[(4, 4), dtype1, "input", "cpu"]
        y: ft.Var[(4, 4), dtype2, "input", "cpu"]
        out: ft.Var[(4, 4), ret_dtype, "output", "cpu"]
        "nid: to_test"
        libop_func(x, y, out)

    x_torch = rand(4, 4, dtype=dtype1)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = rand(4, 4, dtype=dtype2)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_torch = zeros(4, 4, dtype=ret_dtype)
    out_arr = ft.Array(out_torch.numpy(), device)
    f(x_arr, y_arr, out_arr)
    out_torch = torch.tensor(out_arr.numpy())

    assert same(out_torch, torch_func(x_torch, y_torch), dtype=ret_dtype)


@pytest.mark.parametrize('libop_func, torch_func, dtype, ret_dtype', [
    (libop.add, operator.add, "float32", "float32"),
    (libop.sub, operator.sub, "float32", "float32"),
    (libop.mul, operator.mul, "float32", "float32"),
    (libop.truediv, operator.truediv, "float32", "float32"),
    (libop.floordiv, functools.partial(
        torch.div, rounding_mode='floor'), "int32", "int32"),
    (libop.mod, operator.mod, "int32", "int32"),
    (libop.l_and, torch.logical_and, "bool", "bool"),
    (libop.l_or, torch.logical_or, "bool", "bool"),
    (libop.lt, operator.lt, "int32", "bool"),
    (libop.le, operator.le, "int32", "bool"),
    (libop.gt, operator.gt, "int32", "bool"),
    (libop.ge, operator.ge, "int32", "bool"),
    (libop.eq, operator.eq, "int32", "bool"),
    (libop.ne, operator.ne, "int32", "bool"),
])
def test_out_of_place(libop_func, torch_func, dtype, ret_dtype):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4, 4), dtype, "input", "cpu"]
        y: ft.Var[(4, 4), dtype, "input", "cpu"]
        "nid: to_test"
        return libop_func(x, y)

    x_torch = rand(4, 4, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = rand(4, 4, dtype=dtype)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_arr = f(x_arr, y_arr)
    out_torch = torch.tensor(out_arr.numpy())

    assert np.array_equal(out_arr.shape, [4, 4])
    assert same(out_torch, torch_func(x_torch, y_torch), dtype=ret_dtype)


@pytest.mark.parametrize('libop_func, torch_func, dtype, ret_dtype', [
    (operator.add, operator.add, "float32", "float32"),
    (operator.sub, operator.sub, "float32", "float32"),
    (operator.mul, operator.mul, "float32", "float32"),
    (operator.truediv, operator.truediv, "float32", "float32"),
    (operator.floordiv, functools.partial(
        torch.div, rounding_mode='floor'), "int32", "int32"),
    (operator.mod, operator.mod, "int32", "int32"),
    (operator.lt, operator.lt, "int32", "bool"),
    (operator.le, operator.le, "int32", "bool"),
    (operator.gt, operator.gt, "int32", "bool"),
    (operator.ge, operator.ge, "int32", "bool"),
    (operator.eq, operator.eq, "int32", "bool"),
    (operator.ne, operator.ne, "int32", "bool"),
])
def test_operator_overload(libop_func, torch_func, dtype, ret_dtype):
    device = ft.Device(ft.CPU())

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(4, 4), dtype, "input", "cpu"]
        y: ft.Var[(4, 4), dtype, "input", "cpu"]
        "nid: to_test"
        return libop_func(x, y)

    x_torch = rand(4, 4, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy(), device)
    y_torch = rand(4, 4, dtype=dtype)
    y_arr = ft.Array(y_torch.numpy(), device)
    out_arr = f(x_arr, y_arr)
    out_torch = torch.tensor(out_arr.numpy())

    assert np.array_equal(out_arr.shape, [4, 4])
    assert same(out_torch, torch_func(x_torch, y_torch), dtype=ret_dtype)


@pytest.mark.parametrize('libop_func, torch_func, dtype, ret_dtype', [
    (libop.add, operator.add, "float32", "float32"),
    (libop.sub, operator.sub, "float32", "float32"),
    (libop.mul, operator.mul, "float32", "float32"),
    (libop.truediv, operator.truediv, "float32", "float32"),
    (libop.floordiv, functools.partial(
        torch.div, rounding_mode='floor'), "int32", "int32"),
    (libop.mod, operator.mod, "int32", "int32"),
    (libop.lt, operator.lt, "int32", "bool"),
    (libop.le, operator.le, "int32", "bool"),
    (libop.gt, operator.gt, "int32", "bool"),
    (libop.ge, operator.ge, "int32", "bool"),
    (libop.eq, operator.eq, "int32", "bool"),
    (libop.ne, operator.ne, "int32", "bool"),
])
def test_fallback(libop_func, torch_func, dtype, ret_dtype):
    x = rand(4, 4, dtype=dtype)
    y = rand(4, 4, dtype=dtype)
    out = libop_func(x, y)
    assert same(out, torch_func(x, y), dtype=ret_dtype)
