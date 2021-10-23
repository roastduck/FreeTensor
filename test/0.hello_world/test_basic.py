import ir
import numpy as np
import pytest


def test_hello_world():
    with ir.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        x[2, 3] = 2.0
        x[1, 0] = 3.0

    func = ir.lower(ir.Func("main", ["x"], [], ir.pop_ast()), ir.CPU())
    print(func)
    code = ir.codegen(func, ir.CPU())
    print(code)

    x_np = np.zeros((4, 4), dtype="float32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(x=x_arr)
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

    func = ir.lower(ir.Func("main", ["x", "y"], [], ir.pop_ast()), ir.CPU())
    code = ir.codegen(func, ir.CPU())
    print(code)
    x_np = np.array(5, dtype="int32")
    y_np = np.array(0, dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy()

    assert y_np[()] == 11


def test_cast():
    with ir.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = ir.cast(x[()], "int32") * 2

    func = ir.lower(ir.Func("main", ["x", "y"], [], ir.pop_ast()), ir.CPU())
    code = ir.codegen(func, ir.CPU())
    print(code)
    x_np = np.array(2.5, dtype="float32")
    y_np = np.array(0, dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy()

    assert y_np[()] == 4


def test_for():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = x[i] + 1

    func = ir.lower(ir.Func("main", ["x", "y"], [], ir.pop_ast()), ir.CPU())
    code = ir.codegen(func, ir.CPU())
    print(code)
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(x=x_arr, y=y_arr)
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

    func = ir.lower(ir.Func("main", ["y"], [], ir.pop_ast()), ir.CPU())
    code = ir.codegen(func, ir.CPU())
    print(code)
    y_np = np.zeros((4,), dtype="int32")
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(y=y_arr)
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.array([0, 0, 1, 1], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_var_as_shape():
    with ir.VarDef("shape", (2,), "int32", "input", "cpu") as shape:
        with ir.VarDef([("x", shape, "int32", "input", "cpu"),
                        ("y", shape, "int32", "output", "cpu")]) as (x, y):
            with ir.For("i", 0, shape[0]) as i:
                with ir.For("j", 0, shape[1]) as j:
                    y[i, j] = x[i, j] * 2

    func = ir.lower(ir.Func("main", ["shape", "x", "y"], [], ir.pop_ast()),
                    ir.CPU())
    print(func)

    code = ir.codegen(func, ir.CPU())
    print(code)
    shape_np = np.array([4, 4]).astype("int32")
    shape_arr = ir.Array(shape_np, ir.Device(ir.CPU()))
    x_np = np.random.randint(0, 100, (4, 4)).astype("int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_np = np.zeros((4, 4), dtype="int32")
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(shape=shape_arr, x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(4, 4)

    y_std = x_np * 2
    assert np.array_equal(y_np, y_std)


def test_var_as_index():
    with ir.VarDef([("idx", (2,), "int32", "input", "cpu"),
                    ("x", (4, 4), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (idx, x, y):
        y[()] = x[idx]

    func = ir.lower(ir.Func("main", ["idx", "x", "y"], [], ir.pop_ast()),
                    ir.CPU())
    print(func)

    code = ir.codegen(func, ir.CPU())
    print(code)
    idx_np = np.array([1, 2]).astype("int32")
    idx_arr = ir.Array(idx_np, ir.Device(ir.CPU()))
    x_np = np.random.randint(0, 100, (4, 4)).astype("int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_np = np.array(0, dtype="int32")
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    driver.set_params(idx=idx_arr, x=x_arr, y=y_arr)
    driver.run()
    y_np = y_arr.numpy()[0]

    y_std = x_np[1, 2]
    assert np.array_equal(y_np, y_std)


def test_error_missing_parameters():
    with ir.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        x[2, 3] = 2.0
        x[1, 0] = 3.0

    func = ir.lower(ir.Func("main", ["x"], [], ir.pop_ast()), ir.CPU())
    code = ir.codegen(func, ir.CPU())

    driver = ir.Driver(func, code, ir.Device(ir.CPU()))
    with pytest.raises(ir.DriverError):
        driver()
