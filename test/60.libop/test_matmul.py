import torch
import numpy as np

import freetensor as ft
from freetensor import libop


def test_mm():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, y):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        y: ft.Var[(4, 6), "float32", "output", "cpu"]
        #! label: einsum
        libop.matmul_(a, b, y)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy())
    f(a_arr, b_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.matmul(a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bmm_1():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, y):
        a: ft.Var[(2, 4, 5), "float32", "input", "cpu"]
        b: ft.Var[(2, 5, 6), "float32", "input", "cpu"]
        y: ft.Var[(2, 4, 6), "float32", "output", "cpu"]
        #! label: einsum
        libop.matmul_(a, b, y)

    a_torch = torch.rand(2, 4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(2, 5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_torch = torch.zeros(2, 4, 6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy())
    f(a_arr, b_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.matmul(a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bmm_2():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, y):
        a: ft.Var[(2, 4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        y: ft.Var[(2, 4, 6), "float32", "output", "cpu"]
        #! label: einsum
        libop.matmul_(a, b, y)

    a_torch = torch.rand(2, 4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_torch = torch.zeros(2, 4, 6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy())
    f(a_arr, b_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.matmul(a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_mv():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, y):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5,), "float32", "input", "cpu"]
        y: ft.Var[(4,), "float32", "output", "cpu"]
        #! label: einsum
        libop.matmul_(a, b, y)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(5, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_torch = torch.zeros(4, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy())
    f(a_arr, b_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.matmul(a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_vm():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, y):
        a: ft.Var[(5,), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        y: ft.Var[(6,), "float32", "output", "cpu"]
        #! label: einsum
        libop.matmul_(a, b, y)

    a_torch = torch.rand(5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_torch = torch.zeros(6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy())
    f(a_arr, b_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.matmul(a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_scalar_multiply_scalar():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, y):
        a: ft.Var[(), "float32", "input", "cpu"]
        b: ft.Var[(), "float32", "input", "cpu"]
        y: ft.Var[(), "float32", "output", "cpu"]
        #! label: einsum
        libop.matmul_(a, b, y)

    a_arr = ft.Array(np.array(2, dtype="float32"))
    b_arr = ft.Array(np.array(3, dtype="float32"))
    y_arr = ft.Array(np.array(0, dtype="float32"))
    f(a_arr, b_arr, y_arr)

    assert y_arr.numpy() == 6


def test_scalar_multiply_matrix():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b, y):
        a: ft.Var[(), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        y: ft.Var[(5, 6), "float32", "output", "cpu"]
        #! label: einsum
        libop.matmul_(a, b, y)

    a_arr = ft.Array(np.array(2, dtype="float32"))
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_torch = torch.zeros(5, 6, dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy())
    f(a_arr, b_arr, y_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = 2 * b_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_out_of_place():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5,), "float32", "input", "cpu"]
        #! label: gemm
        return libop.matmul(a, b)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(5, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_arr = f(a_arr, b_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.matmul(a_torch, b_torch)
    assert np.array_equal(y_arr.shape, [4])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_operator_overload():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5,), "float32", "input", "cpu"]
        #! label: gemm
        return a @ b

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(5, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_arr = f(a_arr, b_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.matmul(a_torch, b_torch)
    assert np.array_equal(y_arr.shape, [4])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_subtensor():
    device = ft.CPU()

    @ft.optimize(device=device, verbose=1)
    def f(a, b):
        a: ft.Var[(2, 2, 2, 4, 5), "float32", "input", "cpu"]
        b: ft.Var[(2, 2, 2, 5), "float32", "input", "cpu"]
        #! label: gemm
        return a[0, 0, 0] @ b[0, 0, 0]

    a_torch = torch.rand(2, 2, 2, 4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(2, 2, 2, 5, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_arr = f(a_arr, b_arr)
    y_torch = torch.tensor(y_arr.numpy())

    y_std = torch.matmul(a_torch[0, 0, 0], b_torch[0, 0, 0])
    assert np.array_equal(y_arr.shape, [4])
    assert torch.all(torch.isclose(y_torch, y_std))
