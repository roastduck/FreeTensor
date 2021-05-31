import torch
import numpy as np

import ir
import ir.libop


def test_basic():
    device = ir.Device(ir.CPU())

    gemm = ir.libop.gemm("cpu", "float32")

    @ir.transform
    def f(a, b, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (5, 6), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((2,), "int32", "cache", "cpu")
        "nid: gemm"
        gemm([4, 5], [5, 6], y_shape, a, b, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("gemm:V_A_shape")
    s.inline("gemm:V_B_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(4, 6))

    y_std = a_torch @ b_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_trans_A():
    device = ir.Device(ir.CPU())

    gemm = ir.libop.gemm("cpu", "float32", trans_A=True)

    @ir.transform
    def f(a, b, y):
        ir.declare_var(a, (5, 4), "float32", "input", "cpu")
        ir.declare_var(b, (5, 6), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((2,), "int32", "cache", "cpu")
        "nid: gemm"
        gemm([5, 4], [5, 6], y_shape, a, b, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("gemm:V_A_shape")
    s.inline("gemm:V_B_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(5, 4, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(4, 6))

    y_std = a_torch.t() @ b_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_trans_B():
    device = ir.Device(ir.CPU())

    gemm = ir.libop.gemm("cpu", "float32", trans_B=True)

    @ir.transform
    def f(a, b, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (6, 5), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((2,), "int32", "cache", "cpu")
        "nid: gemm"
        gemm([4, 5], [6, 5], y_shape, a, b, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("gemm:V_A_shape")
    s.inline("gemm:V_B_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(6, 5, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(4, 6))

    y_std = a_torch @ b_torch.t()
    assert torch.all(torch.isclose(y_torch, y_std))


def test_trans_AB():
    device = ir.Device(ir.CPU())

    gemm = ir.libop.gemm("cpu", "float32", trans_A=True, trans_B=True)

    @ir.transform
    def f(a, b, y):
        ir.declare_var(a, (5, 4), "float32", "input", "cpu")
        ir.declare_var(b, (6, 5), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((2,), "int32", "cache", "cpu")
        "nid: gemm"
        gemm([5, 4], [6, 5], y_shape, a, b, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("gemm:V_A_shape")
    s.inline("gemm:V_B_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(5, 4, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(6, 5, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(4, 6))

    y_std = a_torch.t() @ b_torch.t()
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias():
    device = ir.Device(ir.CPU())

    gemm = ir.libop.gemm("cpu", "float32", with_bias=True, n_bias_dim=2)

    @ir.transform
    def f(a, b, c, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (5, 6), "float32", "input", "cpu")
        ir.declare_var(c, (4, 6), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((2,), "int32", "cache", "cpu")
        "nid: gemm"
        gemm([4, 5], [5, 6], [4, 6], y_shape, a, b, c, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("gemm:V_A_shape")
    s.inline("gemm:V_B_shape")
    s.inline("gemm:V_C_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    c_torch = torch.rand(4, 6, dtype=torch.float32)
    c_arr = ir.Array(c_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, c_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(4, 6))

    y_std = a_torch @ b_torch + c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias_broadcast_1():
    device = ir.Device(ir.CPU())

    gemm = ir.libop.gemm("cpu", "float32", with_bias=True, n_bias_dim=2)

    @ir.transform
    def f(a, b, c, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (5, 6), "float32", "input", "cpu")
        ir.declare_var(c, (4, 1), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((2,), "int32", "cache", "cpu")
        "nid: gemm"
        gemm([4, 5], [5, 6], [4, 1], y_shape, a, b, c, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("gemm:V_A_shape")
    s.inline("gemm:V_B_shape")
    s.inline("gemm:V_C_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    c_torch = torch.rand(4, 1, dtype=torch.float32)
    c_arr = ir.Array(c_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, c_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(4, 6))

    y_std = a_torch @ b_torch + c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias_broadcast_2():
    device = ir.Device(ir.CPU())

    gemm = ir.libop.gemm("cpu", "float32", with_bias=True, n_bias_dim=1)

    @ir.transform
    def f(a, b, c, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (5, 6), "float32", "input", "cpu")
        ir.declare_var(c, (6,), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((2,), "int32", "cache", "cpu")
        "nid: gemm"
        gemm([4, 5], [5, 6], [6], y_shape, a, b, c, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("gemm:V_A_shape")
    s.inline("gemm:V_B_shape")
    s.inline("gemm:V_C_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    c_torch = torch.rand(6, dtype=torch.float32)
    c_arr = ir.Array(c_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, c_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(4, 6))

    y_std = a_torch @ b_torch + c_torch
    assert torch.all(torch.isclose(y_torch, y_std))


def test_bias_with_coeff():
    device = ir.Device(ir.CPU())

    gemm = ir.libop.gemm("cpu",
                         "float32",
                         with_bias=True,
                         n_bias_dim=2,
                         alpha=2.5,
                         beta=3.8)

    @ir.transform
    def f(a, b, c, y):
        ir.declare_var(a, (4, 5), "float32", "input", "cpu")
        ir.declare_var(b, (5, 6), "float32", "input", "cpu")
        ir.declare_var(c, (4, 6), "float32", "input", "cpu")
        ir.declare_var(y, (4, 6), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((2,), "int32", "cache", "cpu")
        "nid: gemm"
        gemm([4, 5], [5, 6], [4, 6], y_shape, a, b, c, y)

    print(f)
    s = ir.Schedule(f)
    s.inline("gemm:V_A_shape")
    s.inline("gemm:V_B_shape")
    s.inline("gemm:V_C_shape")
    s.inline("y_shape")  # Remove unused output
    f = ir.lower(s.func(), ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ir.Array(a_torch.numpy(), device)
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ir.Array(b_torch.numpy(), device)
    c_torch = torch.rand(4, 6, dtype=torch.float32)
    c_arr = ir.Array(c_torch.numpy(), device)
    y_torch = torch.zeros(4, 6, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(a_arr, b_arr, c_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(4, 6))

    y_std = 2.5 * a_torch @ b_torch + 3.8 * c_torch
    assert torch.all(torch.isclose(y_torch, y_std))
