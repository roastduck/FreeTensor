import freetensor as ft
import numpy as np
import pytest


def test_hello_world():
    with ft.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        x[2, 3] = 2.0
        x[1, 0] = 3.0

    func = ft.lower(ft.Func("main", ["x"], [], ft.pop_ast()), ft.CPU())
    print(func)
    code = ft.codegen(func, ft.CPU())
    print(code)

    x_np = np.zeros((4, 4), dtype="float32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr)
    x_np = x_arr.numpy()

    x_std = np.zeros((4, 4), dtype="float32")
    x_std[2, 3] = 2.0
    x_std[1, 0] = 3.0
    assert np.array_equal(x_np, x_std)


def test_hello_world_float64():
    with ft.VarDef("x", (4, 4), "float64", "output", "cpu") as x:
        x[2, 3] = 2.0
        x[1, 0] = 3.0

    func = ft.lower(ft.Func("main", ["x"], [], ft.pop_ast()), ft.CPU())
    print(func)
    code = ft.codegen(func, ft.CPU())
    print(code)

    x_np = np.zeros((4, 4), dtype="float64")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr)
    x_np = x_arr.numpy()

    x_std = np.zeros((4, 4), dtype="float64")
    x_std[2, 3] = 2.0
    x_std[1, 0] = 3.0
    assert np.array_equal(x_np, x_std)


def test_hello_world_int64():
    with ft.VarDef("x", (4, 4), "int64", "output", "cpu") as x:
        x[2, 3] = 2
        x[1, 0] = 3

    func = ft.lower(ft.Func("main", ["x"], [], ft.pop_ast()), ft.CPU())
    print(func)
    code = ft.codegen(func, ft.CPU())
    print(code)

    x_np = np.zeros((4, 4), dtype="int64")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr)
    x_np = x_arr.numpy()

    x_std = np.zeros((4, 4), dtype="int64")
    x_std[2, 3] = 2
    x_std[1, 0] = 3
    assert np.array_equal(x_np, x_std)


def test_hello_world_bool():
    with ft.VarDef("x", (4, 4), "bool", "output", "cpu") as x:
        x[2, 3] = False
        x[1, 0] = True

    func = ft.lower(ft.Func("main", ["x"], [], ft.pop_ast()), ft.CPU())
    print(func)
    code = ft.codegen(func, ft.CPU())
    print(code)

    x_np = np.zeros((4, 4), dtype="bool")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr)
    x_np = x_arr.numpy()

    x_std = np.zeros((4, 4), dtype="bool")
    x_std[2, 3] = False
    x_std[1, 0] = True
    assert np.array_equal(x_np, x_std)


def test_scalar_op():
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] * 2 + 1

    func = ft.lower(ft.Func("main", ["x", "y"], [], ft.pop_ast()), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    print(code)
    x_np = np.array(5, dtype="int32")
    y_np = np.array(0, dtype="int32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    assert y_np[()] == 11


def test_cast():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = ft.cast(x[()], "int32") * 2

    func = ft.lower(ft.Func("main", ["x", "y"], [], ft.pop_ast()), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    print(code)
    x_np = np.array(2.5, dtype="float32")
    y_np = np.array(0, dtype="int32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    assert y_np[()] == 4


def test_real_div():
    with ft.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[()] = x1[()] / x2[()]

    func = ft.lower(ft.Func("main", ["x1", "x2", "y"], [], ft.pop_ast()),
                    ft.CPU())
    code = ft.codegen(func, ft.CPU())
    print(code)
    x1_np = np.array(5, dtype="int32")
    x2_np = np.array(2, dtype="int32")
    y_np = np.array(0, dtype="float32")
    x1_arr = ft.Array(x1_np, ft.Device(ft.CPU()))
    x2_arr = ft.Array(x2_np, ft.Device(ft.CPU()))
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x1=x1_arr, x2=x2_arr, y=y_arr)
    y_np = y_arr.numpy()

    assert y_np[()] == 2.5


def test_for():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x[i] + 1

    func = ft.lower(ft.Func("main", ["x", "y"], [], ft.pop_ast()), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    print(code)
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_reversed_for():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 3, -1, -1) as i:
            y[i] = x[i] + 1

    func = ft.lower(ft.Func("main", ["x", "y"], [], ft.pop_ast()), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    print(code)
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_if():
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            with ft.If(i < 2):
                y[i] = 0
            with ft.Else():
                y[i] = 1

    func = ft.lower(ft.Func("main", ["y"], [], ft.pop_ast()), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    print(code)
    y_np = np.zeros((4,), dtype="int32")
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([0, 0, 1, 1], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_bool_tensor_as_cond():
    with ft.VarDef([("a", (4,), "bool", "input", "cpu"),
                    ("b", (4,), "bool", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (a, b, y):
        with ft.For("i", 0, 4) as i:
            y[i] = 1
            with ft.If(ft.l_and(a[i], b[i])):
                y[i] = 2

    func = ft.lower(ft.Func("main", ["a", "b", "y"], [], ft.pop_ast()),
                    ft.CPU())
    code = ft.codegen(func, ft.CPU())
    print(code)
    a_np = np.array([False, False, True, True], dtype="bool")
    a_arr = ft.Array(a_np, ft.Device(ft.CPU()))
    b_np = np.array([False, True, False, True], dtype="bool")
    b_arr = ft.Array(b_np, ft.Device(ft.CPU()))
    y_np = np.zeros((4,), dtype="int32")
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(a=a_arr, b=b_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([1, 1, 1, 2], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_var_as_shape():
    with ft.VarDef("shape", (2,), "int32", "input", "cpu") as shape:
        with ft.VarDef([("x", shape, "int32", "input", "cpu"),
                        ("y", shape, "int32", "output", "cpu")]) as (x, y):
            with ft.For("i", 0, shape[0]) as i:
                with ft.For("j", 0, shape[1]) as j:
                    y[i, j] = x[i, j] * 2

    func = ft.lower(ft.Func("main", ["shape", "x", "y"], [], ft.pop_ast()),
                    ft.CPU())
    print(func)

    code = ft.codegen(func, ft.CPU())
    print(code)
    shape_np = np.array([4, 4]).astype("int32")
    shape_arr = ft.Array(shape_np, ft.Device(ft.CPU()))
    x_np = np.random.randint(0, 100, (4, 4)).astype("int32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    y_np = np.zeros((4, 4), dtype="int32")
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(shape=shape_arr,
                                               x=x_arr,
                                               y=y_arr)
    y_np = y_arr.numpy()

    y_std = x_np * 2
    assert np.array_equal(y_np, y_std)


def test_var_as_index():
    with ft.VarDef([("idx", (2,), "int32", "input", "cpu"),
                    ("x", (4, 4), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (idx, x, y):
        y[()] = x[idx]

    func = ft.lower(ft.Func("main", ["idx", "x", "y"], [], ft.pop_ast()),
                    ft.CPU())
    print(func)

    code = ft.codegen(func, ft.CPU())
    print(code)
    idx_np = np.array([1, 2]).astype("int32")
    idx_arr = ft.Array(idx_np, ft.Device(ft.CPU()))
    x_np = np.random.randint(0, 100, (4, 4)).astype("int32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    y_np = np.array(0, dtype="int32")
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(idx=idx_arr, x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = x_np[1, 2]
    assert np.array_equal(y_np, y_std)


def test_error_missing_parameters():
    with ft.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        x[2, 3] = 2.0
        x[1, 0] = 3.0

    func = ft.lower(ft.Func("main", ["x"], [], ft.pop_ast()), ft.CPU())
    code = ft.codegen(func, ft.CPU())

    driver = ft.Driver(func, code, ft.Device(ft.CPU()))
    with pytest.raises(ft.DriverError):
        driver()


def test_inlined_invoke():
    with ft.VarDef("y", (4,), "float32", "output", "cpu") as y:
        y[3] = 2.0
    g = ft.lower(ft.Func("g", ["y"], [], ft.pop_ast()), ft.CPU())

    with ft.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        ft.Invoke(g, x[2])
    f = ft.lower(ft.Func("f", ["x"], [], ft.pop_ast()), ft.CPU())

    print(f)
    code = ft.codegen(f, ft.CPU())
    print(code)

    x_np = np.zeros((4, 4), dtype="float32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    ft.Driver(f, code, ft.Device(ft.CPU()))(x=x_arr)
    x_np = x_arr.numpy()

    x_std = np.zeros((4, 4), dtype="float32")
    x_std[2, 3] = 2.0
    assert np.array_equal(x_np, x_std)


def test_error_modifying_input_tensor():
    with pytest.raises(ft.InvalidProgram):
        with ft.VarDef("x", (4, 4), "float32", "input", "cpu") as x:
            x[2, 3] = 2.0
            x[1, 0] = 3.0
        func = ft.lower(ft.Func("main", ["x"], [], ft.pop_ast()), ft.CPU())


def test_error_modifying_shape_of_a_var_when_using_it():
    with pytest.raises(ft.InvalidProgram):
        with ft.VarDef("n", (), "int32", "inout", "cpu") as n:
            with ft.VarDef([("x", (n[()],), "float32", "input", "cpu"),
                            ("y", (n[()],), "float32", "output", "cpu")
                           ]) as (x, y):
                n[()] = 0  # Error
                with ft.For("i", 0, n[()]) as i:
                    y[i] = x[i] + 1
        func = ft.lower(ft.Func("main", ["n", "x", "y"], [], ft.pop_ast()),
                        ft.CPU())


def test_error_modifying_a_var_when_borrowed_as_a_slice():
    with pytest.raises(ft.InvalidProgram):
        with ft.VarDef([("x", (10, 10), "float32", "input", "cpu"),
                        ("y", (10,), "float32", "output", "cpu"),
                        ("offset", (), "int32", "inout", "cpu")]) as (x, y,
                                                                      offset):
            x_slice = x[offset[()]]  # Borrow
            offset[()] = 0  # Error
            for i in range(10):
                y[i] = x_slice[i]
        func = ft.lower(ft.Func("main", ["x", "y", "offset"], [], ft.pop_ast()),
                        ft.CPU())


def test_target_language_keyword_as_name():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("for", 0, 4) as i:
            y[i] = x[i] + 1

    func = ft.lower(ft.Func("main", ["x", "y"], [], ft.pop_ast()), ft.CPU())
    code = ft.codegen(func, ft.CPU())
    print(code)
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ft.Array(x_np, ft.Device(ft.CPU()))
    y_arr = ft.Array(y_np, ft.Device(ft.CPU()))
    ft.Driver(func, code, ft.Device(ft.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)
