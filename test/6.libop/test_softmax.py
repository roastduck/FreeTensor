import torch
import numpy as np

import ir
import ir.libop


def test_static_shape():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (4, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "output", "cpu")
        "nid: softmax"
        ir.libop.softmax_()(x, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch, torch.softmax(x_torch, axis=-1)))


def test_out_of_place():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y_shape, y):
        ir.declare_var(x, (4, 4), "float32", "input", "cpu")
        ir.declare_var(y_shape, (2,), "int32", "output", "cpu")
        ir.declare_var(y, (4, 4), "float32", "output", "cpu")
        "nid: softmax"
        _y = ir.libop.softmax(axis=-1)(x)
        y_shape[0] = _y.shape(0)
        y_shape[1] = _y.shape(1)
        for i in range(4):
            for j in range(4):
                y[i, j] = _y[i, j]

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_shape_torch = torch.zeros(2, dtype=torch.int32)
    y_shape_arr = ir.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy())

    assert np.array_equal(y_shape_np, [4, 4])
    assert torch.all(torch.isclose(y_torch, torch.softmax(x_torch, axis=-1)))


def test_grad():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (4, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "output", "cpu")
        "nid: softmax"
        ir.libop.softmax_()(x, y)

    print(f)
    f, g, requires, privdes, _ = ir.grad(f, set(["x"]), set(["y"]),
                                         ir.GradTapeMode.NoReuseOnly)
    print("Forward:")
    print(f)
    print("Backward:")
    print(g)
    f = ir.lower(f, ir.CPU())
    print("Forward:")
    print(f)
    g = ir.lower(g, ir.CPU())
    print("Backward:")
    print(g)

    f_code = ir.codegen(f, ir.CPU())
    g_code = ir.codegen(g, ir.CPU())

    def get_shape_and_dtype(func, nid):
        s = ir.Schedule(func)
        vardef = s.find(nid).node()
        shape = []
        for x in vardef.buffer.tensor.shape:
            assert isinstance(x, ir.ffi.IntConst)
            shape.append(x.val)
        if vardef.buffer.tensor.dtype == ir.DataType.Float32:
            dtype = torch.float32
        elif vardef.buffer.tensor.dtype == ir.DataType.Int32:
            dtype = torch.int32
        else:
            assert False
        return shape, dtype

    x_torch = torch.rand(4, 4, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    x_torch.requires_grad = True
    y_torch_ours = torch.zeros(4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch_ours.numpy(), device)
    ir.Driver(f, f_code, device)(x_arr, y_arr)
    y_torch_ours = torch.Tensor(y_arr.numpy())
    y_torch = torch.softmax(x_torch, axis=-1)
    assert torch.all(torch.isclose(y_torch_ours, y_torch))

    y_torch.grad = torch.rand(4, 4, dtype=torch.float32)
    d_y_arr = ir.Array(y_torch.grad.numpy(), device)
    x_grad_torch_ours = torch.zeros(4, 4, dtype=torch.float32)
    d_x_arr = ir.Array(x_grad_torch_ours.numpy(), device)
    kvs = {}
    kvs[privdes['y']] = d_y_arr
    kvs[requires['x']] = d_x_arr
    ir.Driver(g, g_code, device)(x_arr, y_arr, **kvs)
    x_grad_torch_ours = torch.Tensor(d_x_arr.numpy())
    y_torch.backward(y_torch.grad)
    assert torch.all(torch.isclose(x_grad_torch_ours, x_torch.grad, 1e-4, 1e-7))
