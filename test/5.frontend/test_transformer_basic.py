import freetensor as ft
from freetensor import debug
import numpy as np


def test_hello_world():

    def test(x):
        x: ft.Var[(4, 4), "float32", "output", "cpu"]
        x[2, 3] = 2.0
        x[1, 0] = 3.0

    func = ft.lower(ft.transform(test), ft.CPU())
    print(func)
    code = ft.codegen(func, ft.CPU())

    x_np = np.zeros((4, 4), dtype="float32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr)
    x_np = x_arr.numpy()

    x_std = np.zeros((4, 4), dtype="float32")
    x_std[2, 3] = 2.0
    x_std[1, 0] = 3.0
    x_func = np.zeros((4, 4), dtype="float32")
    test(x_func)
    assert np.array_equal(x_np, x_std)
    assert np.array_equal(x_func, x_std)


def test_scalar_op():

    def test(x, y):
        x: ft.Var[(), "int32", "input", "cpu"]
        y: ft.Var[(), "int32", "output", "cpu"]
        y[()] = x[()] * 2 + 1

    func = ft.lower(ft.transform(test), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    x_np = np.array(5, dtype="int32")
    y_np = np.array(0, dtype="int32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.array(0, dtype="int32")
    test(x_np, y_func)

    assert y_np[()] == 11
    assert y_func[()] == 11


def test_return_value_and_runtime_allocation():

    @ft.transform
    def test(x):
        x: ft.Var[(), "int32", "input", "cpu"]
        y = ft.create_var((), "int32", "cpu")
        y[()] = x[()] * 2 + 1
        return y

    print(test)
    func = ft.lower(test, ft.CPU())
    code = ft.codegen(func, ft.CPU())
    print(debug.with_line_no(code))
    x_np = np.array(5, dtype="int32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    y_arr, = ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr)
    y_np = y_arr.numpy()

    assert y_np[()] == 11


def test_for():

    def test(x, y):
        x: ft.Var[(4,), "int32", "input", "cpu"]
        y: ft.Var[(4,), "int32", "output", "cpu"]
        for i in range(0, 4):
            y[i] = x[i] + 1

    func = ft.lower(ft.transform(test), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.zeros((4,), dtype="int32")
    test(x_np, y_func)

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(y_func, y_std)


def test_if():

    def test(y):
        y: ft.Var[(4,), "int32", "output", "cpu"]
        for i in range(0, 4):
            if i < 2:
                y[i] = 0
            else:
                y[i] = 1

    func = ft.lower(ft.transform(test), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    y_np = np.zeros((4,), dtype="int32")
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.zeros((4,), dtype="int32")
    test(y_func)

    y_std = np.array([0, 0, 1, 1], dtype="int32")
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(y_func, y_std)


def test_static_if():

    flag = True

    def test(y):
        y: ft.Var[(4,), "int32", "output", "cpu"]
        for i in range(0, 4):
            value = 0
            if flag:
                value = 1
            y[i] = value

    func = ft.lower(ft.transform(test), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    y_np = np.zeros((4,), dtype="int32")
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.zeros((4,), dtype="int32")
    test(y_func)

    y_std = np.array([1, 1, 1, 1], dtype="int32")
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(y_func, y_std)


def test_static_if_2():

    flag = True

    @ft.inline
    def f(y, value):
        for i in range(0, 4):
            if flag:
                value = 1
            y[i] = value

    @ft.transform
    def test(y):
        y: ft.Var[(4,), "int32", "output", "cpu"]
        f(y, 0)

    func = ft.lower(test, ft.CPU())
    code = ft.codegen(func, ft.CPU())
    y_np = np.zeros((4,), dtype="int32")
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([1, 1, 1, 1], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_for_range():

    def test(x):
        x: ft.Var[(4,), "int32", "output", "cpu"]
        for i in range(4):
            x[i] += 1

    func = ft.lower(ft.transform(test), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr)
    x_np = x_arr.numpy()
    x_func = np.array([1, 2, 3, 4], dtype="int32")
    test(x_func)

    x_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(x_np, x_std)
    assert np.array_equal(x_func, x_std)


def test_std_func_alias():

    def test(x):
        x: ft.Var[(4, 4), "float32", "output", "cpu"]
        x[2, 3] = 2.0
        x[1, 0] = 3.0

    func = ft.lower(ft.transform(test), ft.CPU())
    print(func)
    code = ft.codegen(func, ft.CPU())

    x_np = np.zeros((4, 4), dtype="float32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr)
    x_np = x_arr.numpy()

    x_std = np.zeros((4, 4), dtype="float32")
    x_std[2, 3] = 2.0
    x_std[1, 0] = 3.0
    x_func = np.zeros((4, 4), dtype="float32")
    test(x_func)
    assert np.array_equal(x_np, x_std)
    assert np.array_equal(x_func, x_std)


def test_assert():

    @ft.transform
    def test(x1, x2, y1, y2):
        x1: ft.Var[(4,), "int32", "input", "cpu"]
        x2: ft.Var[(4,), "int32", "input", "cpu"]
        y1: ft.Var[(4,), "int32", "output", "cpu"]
        y2: ft.Var[(4,), "int32", "output", "cpu"]
        for i in range(4):
            y1[i] = x1[i] + x2[i]
            assert x1[i] < x2[i]
            y2[i] = ft.min(x1[i], x2[i])

    print(test)

    with ft.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x1, x2, y1,
                                                                 y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = x1[i] + x2[i]
            with ft.Assert(x1[i] < x2[i]):
                y2[i] = ft.min(x1[i], x2[i])
    std = ft.pop_ast()

    assert std.match(test.body)


def test_immediate_var_return():

    @ft.transform
    def test(x):
        x: ft.Var[(), "int32", "input", "cpu"]
        return ft.var([0, 1, x[()]], "int32", "cpu")

    print(test)
    func = ft.lower(test, ft.CPU())
    code = ft.codegen(func, ft.CPU())
    print(debug.with_line_no(code))
    x_np = np.array(2, dtype="int32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    y_arr, = ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr)
    y_np = y_arr.numpy()

    assert np.all(y_np == np.array([0, 1, 2]))
