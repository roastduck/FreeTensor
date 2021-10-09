import torch
import numpy as np

import ir
import ir.libop


def test_max_pooling_basic():
    device = ir.Device(ir.CPU())

    max_pool_ = ir.libop.max_pool_("cpu", auto_pad='VALID', kernel_shape=[3, 3])

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(y, (2, 3, 12, 12), "float32", "output", "cpu")
        "nid: max_pool"
        max_pool_(x, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 12, 12, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 3, 12, 12))

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[1, 1])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_same_padding():
    device = ir.Device(ir.CPU())

    max_pool_ = ir.libop.max_pool_("cpu",
                                   auto_pad='SAME_UPPER',
                                   kernel_shape=[3, 3])

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(y, (2, 3, 14, 14), "float32", "output", "cpu")
        "nid: y_shape"
        y_shape = ir.create_var((4,), "int32", "cache", "cpu")
        "nid: max_pool"
        max_pool_(x, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 14, 14, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 3, 14, 14))

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           padding=[1, 1],
                                           kernel_size=[3, 3],
                                           stride=[1, 1])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_stride():
    device = ir.Device(ir.CPU())

    max_pool_ = ir.libop.max_pool_("cpu",
                                   auto_pad='VALID',
                                   kernel_shape=[3, 3],
                                   strides=[3, 3])

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (2, 3, 12, 12), "float32", "input", "cpu")
        ir.declare_var(y, (2, 3, 4, 4), "float32", "output", "cpu")
        "nid: max_pool"
        max_pool_(x, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 12, 12, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 4, 4, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 3, 4, 4))

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[3, 3])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_dilation():
    device = ir.Device(ir.CPU())

    max_pool_ = ir.libop.max_pool_("cpu",
                                   auto_pad='VALID',
                                   kernel_shape=[3, 3],
                                   dilations=[2, 2])

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(y, (2, 3, 10, 10), "float32", "output", "cpu")
        "nid: max_pool"
        max_pool_(x, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 10, 10, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 3, 10, 10))

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[1, 1],
                                           dilation=[2, 2])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_max_pooling_out_of_place():
    device = ir.Device(ir.CPU())

    max_pool = ir.libop.max_pool("cpu", auto_pad='VALID', kernel_shape=[3, 3])

    @ir.transform
    def f(x, y_shape, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(y_shape, (4,), "int32", "output", "cpu")
        ir.declare_var(y, (2, 3, 12, 12), "float32", "output", "cpu")
        "nid: max_pool"
        _y = max_pool(x)
        for i in range(4):
            y_shape[i] = _y.shape(i)
        for n in range(2):
            for c in range(3):
                for h in range(12):
                    for w in range(12):
                        y[n, c, h, w] = _y[n, c, h, w]

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_shape_torch = torch.zeros(4, dtype=torch.int32)
    y_shape_arr = ir.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, 12, 12, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 3, 12, 12))

    y_std = torch.nn.functional.max_pool2d(x_torch,
                                           kernel_size=[3, 3],
                                           stride=[1, 1])
    assert np.array_equal(y_shape_np, [2, 3, 12, 12])
    assert torch.all(torch.isclose(y_torch, y_std))


def test_global_avg_pool():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(y, (2, 3), "float32", "output", "cpu")
        "nid: max_pool"
        ir.libop.global_avg_pool_("cpu")(x, y)

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 3))

    y_std = torch.nn.functional.avg_pool2d(x_torch,
                                           kernel_size=[14, 14]).reshape(2, 3)
    assert torch.all(torch.isclose(y_torch, y_std))


def test_global_avg_pool_out_of_place():
    device = ir.Device(ir.CPU())

    @ir.transform
    def f(x, y_shape, y):
        ir.declare_var(x, (2, 3, 14, 14), "float32", "input", "cpu")
        ir.declare_var(y_shape, (2,), "int32", "output", "cpu")
        ir.declare_var(y, (2, 3), "float32", "output", "cpu")
        "nid: max_pool"
        _y = ir.libop.global_avg_pool("cpu")(x)
        for i in range(2):
            y_shape[i] = _y.shape(i)
        for i in range(2):
            for j in range(3):
                y[i, j] = _y[i, j]

    print(f)
    f = ir.lower(f, ir.CPU())
    print(f)

    code = ir.codegen(f, ir.CPU())

    x_torch = torch.rand(2, 3, 14, 14, dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_shape_torch = torch.zeros(2, dtype=torch.int32)
    y_shape_arr = ir.Array(y_shape_torch.numpy(), device)
    y_torch = torch.zeros(2, 3, dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_shape_arr, y_arr)
    y_shape_np = y_shape_arr.numpy()
    y_torch = torch.Tensor(y_arr.numpy().reshape(2, 3))

    y_std = torch.nn.functional.avg_pool2d(x_torch,
                                           kernel_size=[14, 14]).reshape(2, 3)
    assert np.array_equal(y_shape_np, [2, 3])
    assert torch.all(torch.isclose(y_torch, y_std))
