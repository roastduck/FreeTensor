import ir
import ir.debug
import numpy as np


def test_binary_op():

    def test(x, y):
        ir.declare_var(x, (), "int32", "input", "cpu")
        ir.declare_var(y, (), "int32", "output", "cpu")
        y[()] = x[()] * 2 + 1

    func = ir.lower(ir.transform(test), ir.CPU())
    code = ir.codegen(func, ir.CPU())
    x_np = np.array(5, dtype="int32")
    y_np = np.array(0, dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.array(0, dtype="int32")
    test(x_np, y_func)

    assert y_np[()] == 11
    assert y_func[()] == 11


def test_bool_op():

    def test(x, y):
        ir.declare_var(x, (4,), "int32", "input", "cpu")
        ir.declare_var(y, (4,), "int32", "output", "cpu")
        y[0] = (x[0] != 0 and x[1] != 0) or (x[2] != 0 and x[3] != 0)
        y[1] = (x[0] != 0 or x[1] != 0) and (x[2] != 0 or x[2] != 0)
        y[2] = x[0] != 0 and x[1] != 0 and x[2] != 0
        y[3] = x[2] != 0 or x[2] != 0 or x[3] != 0

    func = ir.lower(ir.transform(test), ir.CPU())
    code = ir.codegen(func, ir.CPU())
    x_np = np.array([1, 1, 0, 1], dtype="int32")
    y_np = np.array([0, 0, 0, 0], dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.array([0, 0, 0, 0], dtype="int32")
    test(x_np, y_func)

    y_std = np.array([1, 0, 0, 1], dtype="int32")
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(y_func, y_std)


def test_unary_op():

    def test(x, y):
        ir.declare_var(x, (4,), "int32", "input", "cpu")
        ir.declare_var(y, (4,), "int32", "output", "cpu")
        for i in range(0, 4):
            y[i] = not x[i] != 0

    func = ir.lower(ir.transform(test), ir.CPU())
    code = ir.codegen(func, ir.CPU())
    x_np = np.array([1, 0, 1, 0], dtype="int32")
    y_np = np.array([0, 0, 0, 0], dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.array([0, 0, 0, 0], dtype="int32")
    test(x_np, y_func)

    y_std = np.array([0, 1, 0, 1], dtype="int32")
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(y_func, y_std)


def test_comparison_op():

    def test(x, y):
        ir.declare_var(x, (4,), "int32", "input", "cpu")
        ir.declare_var(y, (4,), "int32", "output", "cpu")
        y[0] = x[0] < x[1] <= x[1] == x[1] != x[2] >= x[2] > x[1]
        y[1] = x[1] != x[1]
        y[2] = x[1] == x[1] != x[1] == x[1]
        y[3] = x[0] == x[0] > x[1]

    func = ir.lower(ir.transform(test), ir.CPU())
    code = ir.codegen(func, ir.CPU())
    x_np = np.array([0, 1, 2, 3], dtype="int32")
    y_np = np.array([1, 0, 0, 0], dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.array([0, 0, 0, 0], dtype="int32")
    test(x_np, y_func)

    y_std = np.array([1, 0, 0, 0], dtype="int32")
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(y_func, y_std)


def test_if_expr():

    def test(x1, x2, y):
        ir.declare_var(x1, (), "int32", "input", "cpu")
        ir.declare_var(x2, (), "int32", "input", "cpu")
        ir.declare_var(y, (), "int32", "output", "cpu")
        y[()] = x1[()] if x1[()] > 2 * x2[()] else x2[()]

    func = ir.lower(ir.transform(test), ir.CPU())
    code = ir.codegen(func, ir.CPU())
    print(ir.debug.with_line_no(code))
    x1_np = np.array(5, dtype="int32")
    x2_np = np.array(2, dtype="int32")
    y_np = np.array(0, dtype="int32")
    x1_arr = ir.Array(x1_np, ir.Device(ir.CPU()))
    x2_arr = ir.Array(x2_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(x1=x1_arr, x2=x2_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.array(0, dtype="int32")
    test(x1_np, x2_np, y_func)

    y_std = np.array(5, dtype="int32")
    assert y_np[0] == y_std[()]
    assert y_func[()] == y_std[()]
