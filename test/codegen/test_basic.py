import ir
import numpy as np


def test_hello_world():
    with ir.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        x[2, 3] = 2.0
        x[1, 0] = 3.0

    ast = ir.lower(ir.pop_ast(), ir.CPU())
    print(ast)
    code, params = ir.codegen(ast, ir.CPU())
    print(code)

    x_np = np.zeros((4, 4), dtype="float32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    driver = ir.Driver(code, params, ir.Device(ir.CPU()))
    driver.set_params({"x": x_arr})
    driver.run()
    x_np = x_arr.numpy().reshape(4, 4)

    x_std = np.zeros((4, 4), dtype="float32")
    x_std[2, 3] = 2.0
    x_std[1, 0] = 3.0
    assert np.array_equal(x_np, x_std)


def test_scalar_op():
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] * 2 + 1

    code, params = ir.codegen(ir.lower(ir.pop_ast(), ir.CPU()), ir.CPU())
    print(code)
    x_np = np.array(5, dtype="int32")
    y_np = np.array(0, dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    driver = ir.Driver(code, params, ir.Device(ir.CPU()))
    driver.set_params({"x": x_arr, "y": y_arr})
    driver.run()
    y_np = y_arr.numpy()

    assert y_np[()] == 11


def test_for():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = x[i] + 1

    code, params = ir.codegen(ir.lower(ir.pop_ast(), ir.CPU()), ir.CPU())
    print(code)
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    driver = ir.Driver(code, params, ir.Device(ir.CPU()))
    driver.set_params({"x": x_arr, "y": y_arr})
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_if():
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.If(i < 2):
                y[i] = 0
            with ir.Else():
                y[i] = 1

    code, params = ir.codegen(ir.lower(ir.pop_ast(), ir.CPU()), ir.CPU())
    print(code)
    y_np = np.zeros((4,), dtype="int32")
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    driver = ir.Driver(code, params, ir.Device(ir.CPU()))
    driver.set_params({"y": y_arr})
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.array([0, 0, 1, 1], dtype="int32")
    assert np.array_equal(y_np, y_std)
