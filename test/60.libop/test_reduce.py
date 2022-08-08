import torch
import pytest
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


@pytest.mark.parametrize('libop_func, torch_func, dtype', [
    (libop.reduce_sum_, torch.sum, "float32"),
    (libop.reduce_prod_, torch.prod, "float32"),
    (libop.reduce_max_, lambda *args, **kvs: torch.max(*args, **kvs).values,
     "float32"),
    (libop.reduce_min_, lambda *args, **kvs: torch.min(*args, **kvs).values,
     "float32"),
    (libop.all_, torch.all, "bool"),
    (libop.any_, torch.any, "bool"),
])
def test_static(libop_func, torch_func, dtype):
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(3, 4, 5), dtype, "input", "cpu"]
        y: ft.Var[(3, 5), dtype, "output", "cpu"]
        #! nid: reduce
        libop_func(x, y, axes=[1], keepdims=False)

    x_torch = rand(3, 4, 5, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy())
    y_torch = zeros(3, 5, dtype=dtype)
    y_arr = ft.Array(y_torch.numpy())
    f(x_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert same(y_torch, torch_func(x_torch, axis=1), dtype=dtype)


@pytest.mark.parametrize('libop_func, torch_func, dtype', [
    (libop.reduce_sum_, torch.sum, "float32"),
    (libop.reduce_prod_, torch.prod, "float32"),
    (libop.reduce_max_, lambda *args, **kvs: torch.max(*args, **kvs).values,
     "float32"),
    (libop.reduce_min_, lambda *args, **kvs: torch.min(*args, **kvs).values,
     "float32"),
    (libop.all_, torch.all, "bool"),
    (libop.any_, torch.any, "bool"),
])
def test_keepdims(libop_func, torch_func, dtype):
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x, y):
        x: ft.Var[(3, 4, 5), dtype, "input", "cpu"]
        y: ft.Var[(3, 1, 5), dtype, "output", "cpu"]
        #! nid: reduce
        libop_func(x, y, axes=[1], keepdims=True)

    x_torch = rand(3, 4, 5, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy())
    y_torch = zeros(3, 1, 5, dtype=dtype)
    y_arr = ft.Array(y_torch.numpy())
    f(x_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert same(y_torch, torch_func(x_torch, axis=1, keepdim=True), dtype=dtype)


@pytest.mark.parametrize('libop_func, torch_func, dtype', [
    (libop.reduce_sum, torch.sum, "float32"),
    (libop.reduce_prod, torch.prod, "float32"),
    (libop.reduce_max, lambda *args, **kvs: torch.max(*args, **kvs).values,
     "float32"),
    (libop.reduce_min, lambda *args, **kvs: torch.min(*args, **kvs).values,
     "float32"),
    (libop.all, torch.all, "bool"),
    (libop.any, torch.any, "bool"),
])
def test_out_of_place(libop_func, torch_func, dtype):
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x):
        x: ft.Var[(3, 4, 5), dtype, "input", "cpu"]
        #! nid: reduce
        return libop_func(x, axes=[1], keepdims=False)

    x_torch = rand(3, 4, 5, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy())
    y_arr = f(x_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert np.array_equal(y_arr.shape, [3, 5])
    assert same(y_torch, torch_func(x_torch, axis=1), dtype=dtype)


@pytest.mark.parametrize('libop_func, torch_func, dtype', [
    (libop.reduce_sum, torch.sum, "float32"),
    (libop.reduce_prod, torch.prod, "float32"),
    (libop.reduce_max, lambda *args, **kvs: torch.max(*args, **kvs).values,
     "float32"),
    (libop.reduce_min, lambda *args, **kvs: torch.min(*args, **kvs).values,
     "float32"),
    (libop.all, torch.all, "bool"),
    (libop.any, torch.any, "bool"),
])
def test_out_of_place_keepdims(libop_func, torch_func, dtype):
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(x):
        x: ft.Var[(3, 4, 5), dtype, "input", "cpu"]
        #! nid: reduce
        return libop_func(x, axes=[1], keepdims=True)

    x_torch = rand(3, 4, 5, dtype=dtype)
    x_arr = ft.Array(x_torch.numpy())
    y_arr = f(x_arr)
    y_torch = torch.tensor(y_arr.numpy())

    assert np.array_equal(y_arr.shape, [3, 1, 5])
    assert same(y_torch, torch_func(x_torch, axis=1, keepdim=True), dtype=dtype)
