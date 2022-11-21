import freetensor as ft
from freetensor import debug
import numpy as np


def test_binary_op():

    def test(x, y):
        x: ft.Var[(), "int32", "input", "cpu"]
        y: ft.Var[(), "int32", "output", "cpu"]
        y[()] = x[()] * 2 + 1

    func = ft.lower(ft.transform(test), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    x_np = np.array(5, dtype="int32")
    y_np = np.array(0, dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.Driver(func, code, ft.CPU())(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.array(0, dtype="int32")
    test(x_np, y_func)

    assert y_np[()] == 11
    assert y_func[()] == 11


def test_bool_op():

    def test(x, y):
        x: ft.Var[(4,), "int32", "input", "cpu"]
        y: ft.Var[(4,), "bool", "output", "cpu"]
        y[0] = (x[0] != 0 and x[1] != 0) or (x[2] != 0 and x[3] != 0)
        y[1] = (x[0] != 0 or x[1] != 0) and (x[2] != 0 or x[2] != 0)
        y[2] = x[0] != 0 and x[1] != 0 and x[2] != 0
        y[3] = x[2] != 0 or x[2] != 0 or x[3] != 0

    func = ft.lower(ft.transform(test), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    x_np = np.array([1, 1, 0, 1], dtype="int32")
    y_np = np.array([0, 0, 0, 0], dtype="bool")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.Driver(func, code, ft.CPU())(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.array([0, 0, 0, 0], dtype="bool")
    test(x_np, y_func)

    y_std = np.array([1, 0, 0, 1], dtype="bool")
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(y_func, y_std)


def test_unary_op():

    def test(x, y):
        x: ft.Var[(4,), "int32", "input", "cpu"]
        y: ft.Var[(4,), "bool", "output", "cpu"]
        for i in range(0, 4):
            y[i] = not x[i] != 0

    func = ft.lower(ft.transform(test), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    x_np = np.array([1, 0, 1, 0], dtype="int32")
    y_np = np.array([0, 0, 0, 0], dtype="bool")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.Driver(func, code, ft.CPU())(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.array([0, 0, 0, 0], dtype="bool")
    test(x_np, y_func)

    y_std = np.array([0, 1, 0, 1], dtype="bool")
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(y_func, y_std)


def test_comparison_op():

    def test(x, y):
        x: ft.Var[(4,), "int32", "input", "cpu"]
        y: ft.Var[(4,), "bool", "output", "cpu"]
        y[0] = x[0] < x[1] <= x[1] == x[1] != x[2] >= x[2] > x[1]
        y[1] = x[1] != x[1]
        y[2] = x[1] == x[1] != x[1] == x[1]
        y[3] = x[0] == x[0] > x[1]

    func = ft.lower(ft.transform(test), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    x_np = np.array([0, 1, 2, 3], dtype="int32")
    y_np = np.array([1, 0, 0, 0], dtype="bool")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.Driver(func, code, ft.CPU())(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.array([0, 0, 0, 0], dtype="bool")
    test(x_np, y_func)

    y_std = np.array([1, 0, 0, 0], dtype="bool")
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(y_func, y_std)


def test_if_expr():

    def test(x1, x2, y):
        x1: ft.Var[(), "int32", "input", "cpu"]
        x2: ft.Var[(), "int32", "input", "cpu"]
        y: ft.Var[(), "int32", "output", "cpu"]
        y[()] = x1[()] if x1[()] > 2 * x2[()] else x2[()]

    func = ft.lower(ft.transform(test), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    print(debug.with_line_no(code))
    x1_np = np.array(5, dtype="int32")
    x2_np = np.array(2, dtype="int32")
    y_np = np.array(0, dtype="int32")
    x1_arr = ft.Array(x1_np)
    x2_arr = ft.Array(x2_np)
    y_arr = ft.Array(y_np)
    ft.Driver(func, code, ft.CPU())(x1=x1_arr, x2=x2_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.array(0, dtype="int32")
    test(x1_np, x2_np, y_func)

    y_std = np.array(5, dtype="int32")
    assert y_np[()] == y_std[()]
    assert y_func[()] == y_std[()]
