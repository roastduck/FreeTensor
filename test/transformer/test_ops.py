import ir
import numpy as np


def test_binary_op():

    def test(x, y):
        ir.declare_var(x, (), "int32", "input", "cpu")
        ir.declare_var(y, (), "int32", "output", "cpu")
        y[()] = x[()] * 2 + 1

    code, params = ir.codegen(ir.lower(ir.transform(test), ir.CPU()), ir.CPU())
    # print(code)
    x_np = np.array(5, dtype="int32")
    y_np = np.array(0, dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    driver = ir.Driver(code, params, ir.Device(ir.CPU()))
    driver.set_params({"x": x_arr, "y": y_arr})
    driver.run()
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

    code, params = ir.codegen(ir.lower(ir.transform(test), ir.CPU()), ir.CPU())
    # print(code)
    x_np = np.array([1, 1, 0, 1], dtype="int32")
    y_np = np.array([0, 0, 0, 0], dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    driver = ir.Driver(code, params, ir.Device(ir.CPU()))
    driver.set_params({"x": x_arr, "y": y_arr})
    driver.run()
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

    code, params = ir.codegen(ir.lower(ir.transform(test), ir.CPU()), ir.CPU())
    # print(code)
    x_np = np.array([1, 0, 1, 0], dtype="int32")
    y_np = np.array([0, 0, 0, 0], dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    driver = ir.Driver(code, params, ir.Device(ir.CPU()))
    driver.set_params({"x": x_arr, "y": y_arr})
    driver.run()
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

    code, params = ir.codegen(ir.lower(ir.transform(test), ir.CPU()), ir.CPU())
    # print(code)
    x_np = np.array([0, 1, 2, 3], dtype="int32")
    y_np = np.array([1, 0, 0, 0], dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    driver = ir.Driver(code, params, ir.Device(ir.CPU()))
    driver.set_params({"x": x_arr, "y": y_arr})
    driver.run()
    y_np = y_arr.numpy()
    y_func = np.array([0, 0, 0, 0], dtype="int32")
    test(x_np, y_func)

    y_std = np.array([1, 0, 0, 0], dtype="int32")
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(y_func, y_std)
