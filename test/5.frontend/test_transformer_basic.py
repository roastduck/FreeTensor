import ir
import ir.debug
import numpy as np


def test_hello_world():

    def test(x):
        ir.declare_var(x, (4, 4), "float32", "output", "cpu")
        x[2, 3] = 2.0
        x[1, 0] = 3.0

    func = ir.lower(ir.transform(test), ir.CPU())
    print(func)
    code = ir.codegen(func, ir.CPU())

    x_np = np.zeros((4, 4), dtype="float32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(x=x_arr)
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


def test_return_value_and_runtime_allocation():

    @ir.transform
    def test(x):
        ir.declare_var(x, (), "int32", "input", "cpu")
        y = ir.create_var((), "int32", "cpu")
        y[()] = x[()] * 2 + 1
        return y

    print(test)
    func = ir.lower(test, ir.CPU())
    code = ir.codegen(func, ir.CPU())
    print(ir.debug.with_line_no(code))
    x_np = np.array(5, dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr, = ir.Driver(func, code, ir.Device(ir.CPU()))(x=x_arr)
    y_np = y_arr.numpy()

    assert y_np[()] == 11


def test_for():

    def test(x, y):
        ir.declare_var(x, (4,), "int32", "input", "cpu")
        ir.declare_var(y, (4,), "int32", "output", "cpu")
        for i in range(0, 4):
            y[i] = x[i] + 1

    func = ir.lower(ir.transform(test), ir.CPU())
    code = ir.codegen(func, ir.CPU())
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.zeros((4,), dtype="int32")
    test(x_np, y_func)

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(y_func, y_std)


def test_if():

    def test(y):
        ir.declare_var(y, (4,), "int32", "output", "cpu")
        for i in range(0, 4):
            if i < 2:
                y[i] = 0
            else:
                y[i] = 1

    func = ir.lower(ir.transform(test), ir.CPU())
    code = ir.codegen(func, ir.CPU())
    y_np = np.zeros((4,), dtype="int32")
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.zeros((4,), dtype="int32")
    test(y_func)

    y_std = np.array([0, 0, 1, 1], dtype="int32")
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(y_func, y_std)


def test_for_range():

    def test(x):
        ir.declare_var(x, (4,), "int32", "output", "cpu")
        for i in range(4):
            x[i] += 1

    func = ir.lower(ir.transform(test), ir.CPU())
    code = ir.codegen(func, ir.CPU())
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(x=x_arr)
    x_np = x_arr.numpy()
    x_func = np.array([1, 2, 3, 4], dtype="int32")
    test(x_func)

    x_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(x_np, x_std)
    assert np.array_equal(x_func, x_std)


def test_std_func_alias():

    def test(x):
        ir.core.declare_var(x, (4, 4), "float32", "output", "cpu")
        x[2, 3] = 2.0
        x[1, 0] = 3.0

    func = ir.lower(ir.transform(test), ir.CPU())
    print(func)
    code = ir.codegen(func, ir.CPU())

    x_np = np.zeros((4, 4), dtype="float32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(x=x_arr)
    x_np = x_arr.numpy()

    x_std = np.zeros((4, 4), dtype="float32")
    x_std[2, 3] = 2.0
    x_std[1, 0] = 3.0
    x_func = np.zeros((4, 4), dtype="float32")
    test(x_func)
    assert np.array_equal(x_np, x_std)
    assert np.array_equal(x_func, x_std)


def test_assert():

    @ir.transform
    def test(x, y):
        ir.core.declare_var(x, (10,), "int32", "input", "cpu")
        ir.core.declare_var(y, (10,), "int32", "output", "cpu")
        y[0] = 0
        y[1] = 1
        for i in range(2, 10):
            y[i] = y[i - 1] + y[i - 2]
            assert (x[i] == y[i])
        assert (y[9] == 34)

    func = ir.lower(test, ir.CPU())
    code = ir.codegen(func, ir.CPU())
    print(code)
    x_np = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34], dtype="int32")
    x = ir.Array(x_np, ir.Device(ir.CPU()))
    y = ir.Array(np.zeros((10,), dtype="int32"), ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(x=x, y=y)
    y_np = y.numpy()
    assert np.array_equal(x_np, y_np)


def test_immediate_var_return():

    @ir.transform
    def test(x):
        ir.declare_var(x, (), "int32", "input", "cpu")
        return ir.var([0, 1, x[()]], "int32", "cpu")

    print(test)
    func = ir.lower(test, ir.CPU())
    code = ir.codegen(func, ir.CPU())
    print(ir.debug.with_line_no(code))
    x_np = np.array(2, dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr, = ir.Driver(func, code, ir.Device(ir.CPU()))(x=x_arr)
    y_np = y_arr.numpy()

    assert np.all(y_np == np.array([0, 1, 2]))
