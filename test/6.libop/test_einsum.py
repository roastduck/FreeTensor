import torch
import numpy as np

import ir
import ir.libop


def test_basic():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(a, b, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (5,), "float32", "input", "cpu")
        ir.declare_var(y, (4,), "float32", "output", "cpu")
        "nid: einsum"
        ir.libop.einsum_("ij,j->i")(a, b, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.einsum("ij,j->i", a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_broadcast():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(a, b, y):
        ir.declare_var(a, (4, 1), "float32", "input", "cpu")
        ir.declare_var(b, (5,), "float32", "input", "cpu")
        ir.declare_var(y, (4,), "float32", "output", "cpu")
        "nid: einsum"
        ir.libop.einsum_("ij,j->i")(a, b, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 1, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.einsum("ij,j->i", a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_out_of_place():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(a, b, y_shape, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (5,), "float32", "input", "cpu")
        ir.declare_var(y_shape, (1,), "int32", "output", "cpu")
        ir.declare_var(y, (4,), "float32", "output", "cpu")
        "nid: gemm"
        _y = ir.libop.einsum("ij,j->i")(a, b)
        y_shape[0] = _y.shape(0)
        for i in range(4):
            y[i] = _y[i]

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    y_shape_torch = torch.zeros(1, dtype=torch.int32)
    y_shape_arr = ir.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy())

    y_std = torch.einsum("ij,j->i", a_torch, b_torch)
    assert np.array_equal(y_shape_np, [4])
    assert torch.all(torch.isclose(y_torch, y_std))
