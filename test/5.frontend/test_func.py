import ir
import ir.debug
import math
import functools
import pytest
import numpy as np

# TODO: Currently, a callee function must be in the global scope. Can we support a local scope?


@ir.inline
def g_global(y):
    '''nid: S0'''
    y[0] = 2.0
    '''nid: S1'''
    y[1] = 3.0


@ir.transform
def f_global(y):
    ir.declare_var(y, (2,), "float32", "output", "cpu")
    g_global(y)


def test_basic_call():

    @ir.inline
    def g(y):
        '''nid: S0'''
        y[0] = 2.0
        '''nid: S1'''
        y[1] = 3.0

    @ir.transform
    def f(y):
        ir.declare_var(y, (2,), "float32", "output", "cpu")
        g(y)

    func = ir.lower(f, ir.CPU())
    print(func)

    with ir.VarDef("y", (2,), "float32", "output", "cpu") as y:
        y[0] = 2.0
        y[1] = 3.0
    std = ir.pop_ast()
    assert std.match(func.body)


def test_global_functions():

    func = ir.lower(f_global, ir.CPU())
    print(func)

    with ir.VarDef("y", (2,), "float32", "output", "cpu") as y:
        y[0] = 2.0
        y[1] = 3.0
    std = ir.pop_ast()
    assert std.match(func.body)


def test_called_multiple_times():

    @ir.inline
    def g(y):
        '''nid: S0'''
        y[0] = 2.0
        '''nid: S1'''
        y[1] = 3.0

    @ir.transform
    def f(y1, y2):
        ir.declare_var(y1, (2,), "float32", "output", "cpu")
        ir.declare_var(y2, (2,), "float32", "output", "cpu")
        '''nid: C1'''
        g(y1)
        '''nid: C2'''
        g(y2)

    func = ir.lower(f, ir.CPU())
    print(func)

    with ir.VarDef([("y1", (2,), "float32", "output", "cpu"),
                    ("y2", (2,), "float32", "output", "cpu")]) as (y1, y2):
        y1[0] = 2.0
        y1[1] = 3.0
        y2[0] = 2.0
        y2[1] = 3.0
    std = ir.pop_ast()
    assert std.match(func.body)

    s = ir.Schedule(func)
    assert len(s.find_all("C1->S0")) == 1
    assert len(s.find_all("C1->S1")) == 1
    assert len(s.find_all("C2->S0")) == 1
    assert len(s.find_all("C2->S1")) == 1


def test_call_with_external_data():
    data = ir.Array(np.array([[0, 1], [2, 3]], dtype=np.int32),
                    ir.Device(ir.CPU()))

    @ir.inline
    def g(x, y):
        for i in range(2):
            for j in range(2):
                y[i, j] = x[i, j] * 2

    @ir.transform
    def f(y):
        ir.declare_var(y, (2, 2), "int32", "output", "cpu")
        g(ir.capture_var(data), y)

    func = ir.lower(f, ir.CPU())
    print(func)

    with ir.VarDef("y", (2, 2), "int32", "output", "cpu") as y:
        with ir.VarDef("x", (2, 2), "int32", "input", "cpu") as x:
            with ir.For("i", 0, 2) as i:
                with ir.For("j", 0, 2) as j:
                    y[i, j] = x[i, j] * 2
    std = ir.pop_ast()
    assert std.match(func.body)

    code = ir.codegen(func, ir.CPU())
    print(ir.debug.with_line_no(code))

    y_np = np.zeros((2, 2), dtype="int32")
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(y=y_arr)
    y_np = y_arr.numpy()

    assert np.array_equal(y_np, data.numpy() * 2)


def test_call_with_literal_data():
    dev = ir.Device(ir.CPU())

    @ir.inline
    def g(x, y):
        for i in range(2):
            for j in range(2):
                y[i, j] = x[i, j] * 2

    @ir.transform
    def f(y):
        ir.declare_var(y, (2, 2), "int32", "output", "cpu")
        g(
            ir.capture_var(
                ir.Array(np.array([[0, 1], [2, 3]], dtype=np.int32), dev)), y)

    func = ir.lower(f, ir.CPU())
    print(func)

    with ir.VarDef("y", (2, 2), "int32", "output", "cpu") as y:
        with ir.VarDef("x", (2, 2), "int32", "input", "cpu") as x:
            with ir.For("i", 0, 2) as i:
                with ir.For("j", 0, 2) as j:
                    y[i, j] = x[i, j] * 2
    std = ir.pop_ast()
    assert std.match(func.body)

    code = ir.codegen(func, ir.CPU())
    print(ir.debug.with_line_no(code))

    y_np = np.zeros((2, 2), dtype="int32")
    y_arr = ir.Array(y_np, dev)
    ir.Driver(func, code, dev)(y=y_arr)
    y_np = y_arr.numpy()

    assert np.array_equal(y_np, np.array([[0, 1], [2, 3]], dtype=np.int32) * 2)


def test_call_with_fixed_dim_at_front():

    @ir.inline
    def g(x1, x2, y):
        for i in range(4):
            y[i] = x1[i] + x2[i]

    @ir.transform
    def f(x1, x2, y):
        ir.declare_var(x1, (4, 4), "float32", "input", "cpu")
        ir.declare_var(x2, (4, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "output", "cpu")
        for i in range(4):
            g(x1[i], x2[i], y[i])

    func = ir.lower(f, ir.CPU())
    print(func)

    with ir.VarDef([("x1", (4, 4), "float32", "input", "cpu"),
                    ("x2", (4, 4), "float32", "input", "cpu"),
                    ("y", (4, 4), "float32", "output", "cpu")]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                y[i, j] = x1[i, j] + x2[i, j]
    std = ir.pop_ast()
    assert std.match(func.body)


def test_call_with_fixed_dim_at_back():

    @ir.inline
    def g(x1, x2, y):
        for i in range(4):
            y[i] = x1[i] + x2[i]

    @ir.transform
    def f(x1, x2, y):
        ir.declare_var(x1, (4, 4), "float32", "input", "cpu")
        ir.declare_var(x2, (4, 4), "float32", "input", "cpu")
        ir.declare_var(y, (4, 4), "float32", "output", "cpu")
        for i in range(4):
            g(x1[:, i], x2[:, i], y[:, i])

    func = ir.lower(f, ir.CPU())
    print(func)

    with ir.VarDef([("x1", (4, 4), "float32", "input", "cpu"),
                    ("x2", (4, 4), "float32", "input", "cpu"),
                    ("y", (4, 4), "float32", "output", "cpu")]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                y[j, i] = x1[j, i] + x2[j, i]
    std = ir.pop_ast()
    assert std.match(func.body)


def test_call_with_slice():

    @ir.inline
    def g(x1, x2, y):
        for i in range(4):
            y[i] = x1[i] + x2[i]

    @ir.transform
    def f(x1, x2, y):
        ir.declare_var(x1, (5,), "float32", "input", "cpu")
        ir.declare_var(x2, (5,), "float32", "input", "cpu")
        ir.declare_var(y, (5,), "float32", "output", "cpu")
        y[0] = 0.
        g(x1[1:], x2[1:], y[1:])

    func = ir.lower(f, ir.CPU())
    print(func)

    with ir.VarDef([("x1", (5,), "float32", "input", "cpu"),
                    ("x2", (5,), "float32", "input", "cpu"),
                    ("y", (5,), "float32", "output", "cpu")]) as (x1, x2, y):
        y[0] = 0.
        with ir.For("i", 0, 4) as i:
            y[i + 1] = x1[i + 1] + x2[i + 1]
    std = ir.pop_ast()
    assert std.match(func.body)


def test_call_with_scalar():

    @ir.inline
    def g(x1, x2, y):
        y[()] = x1[()] + x2[()]

    @ir.transform
    def f(x1, x2, y):
        ir.declare_var(x1, (4,), "float32", "input", "cpu")
        ir.declare_var(x2, (4,), "float32", "input", "cpu")
        ir.declare_var(y, (4,), "float32", "output", "cpu")
        for i in range(4):
            g(x1[i], x2[i], y[i])

    func = ir.lower(f, ir.CPU())
    print(func)

    with ir.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            y[i] = x1[i] + x2[i]
    std = ir.pop_ast()
    assert std.match(func.body)


def test_call_with_literal_scalar():

    @ir.inline
    def g(x1, x2, y):
        y[()] = x1 + x2

    @ir.transform
    def f(y):
        ir.declare_var(y, (4,), "float32", "output", "cpu")
        for i in range(4):
            g(1, 2, y[i])

    func = ir.lower(f, ir.CPU())
    print(func)

    with ir.VarDef("y", (4,), "float32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = 3
    std = ir.pop_ast()
    assert std.match(func.body)


def test_external_call():

    @ir.transform
    def func(y):
        ir.declare_var(y, (), "int32", "output", "cpu")
        y[()] = math.gcd(10, 15)

    with ir.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 5
    std = ir.pop_ast()
    assert std.match(func.body)


def test_use_external_call_to_build_runtime_ops():

    @ir.inline
    def g(x1, x2, x3, y):
        y[()] = functools.reduce(lambda x, y: x * y, [x1, x2, x3])

    @ir.transform
    def f(x1, x2, x3, y):
        ir.declare_var(x1, (), "int32", "input", "cpu")
        ir.declare_var(x2, (), "int32", "input", "cpu")
        ir.declare_var(x3, (), "int32", "input", "cpu")
        ir.declare_var(y, (), "int32", "output", "cpu")
        g(x1, x2, x3, y)

    func = ir.lower(f, ir.CPU())
    print(func)

    with ir.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("x3", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x1, x2, x3, y):
        y[()] = x1[()] * x2[()] * x3[()]
    std = ir.pop_ast()
    assert std.match(func.body)


def test_error_missing_parameters():

    @ir.inline
    def g(y):
        '''nid: S0'''
        y[0] = 2.0
        '''nid: S1'''
        y[1] = 3.0

    def f(y):
        ir.declare_var(y, (2,), "float32", "output", "cpu")
        g()

    with pytest.raises(ir.StagingError):
        ir.transform(f)


def test_return():
    dev = ir.Device(ir.CPU())

    @ir.inline
    def test_i(a, b):
        c = ir.create_var((2, 2), "int32", "cpu")
        d = ir.create_var((2, 2), "int32", "cpu")
        for i in range(2):
            for j in range(2):
                b[i, j] = a[i, j]
                c[i, j] = b[i, j] * a[i, j]
                d[i, j] = b[i, j] + a[i, j]
        return c, d

    @ir.transform
    def test(y, c, d):
        ir.declare_var(y, (2, 2), "int32", "output", "cpu")
        ir.declare_var(c, (2, 2), "int32", "output", "cpu")
        ir.declare_var(d, (2, 2), "int32", "output", "cpu")
        c1, d1 = test_i(
            ir.capture_var(
                ir.Array(np.array([[1, 2], [3, 4]], dtype=np.int32), dev)), y)
        for i in range(2):
            for j in range(2):
                c[i, j] = c1[i, j]
                d[i, j] = d1[i, j]

    func = ir.lower(test, ir.CPU())
    print(func)

    with ir.VarDef([("y", (2, 2), "int32", "output", "cpu"),
                    ("c", (2, 2), "int32", "output", "cpu"),
                    ("d", (2, 2), "int32", "output", "cpu")]) as (y, c, d):
        with ir.VarDef("a", (2, 2), "int32", "input", "cpu") as a:
            with ir.VarDef([("c1", (2, 2), "int32", "cache", "cpu"),
                            ("d1", (2, 2), "int32", "cache", "cpu")]) as (c1,
                                                                          d1):
                with ir.For("i", 0, 2) as i:
                    with ir.For("j", 0, 2) as j:
                        y[i, j] = a[i, j]
                        c1[i, j] = y[i, j] * a[i, j]
                        d1[i, j] = y[i, j] + a[i, j]
                with ir.For("i1", 0, 2) as i:
                    with ir.For("j1", 0, 2) as j:
                        c[i, j] = c1[i, j]
                        d[i, j] = d1[i, j]
    std = ir.pop_ast()
    print(std)
    assert std.match(func.body)


def test_return_returned_value():

    @ir.inline
    def h(x):
        y1 = ir.create_var((4,), "int32", "cpu")
        y2 = ir.create_var((4,), "int32", "cpu")
        for i in range(4):
            y1[i] = x[i]
            y2[i] = x[i * 2]
        return y1, y2

    @ir.inline
    def g(x):
        y1, y2 = h(x)
        return y2, y1

    @ir.transform
    def f(x, w1, w2):
        ir.declare_var(x, (8,), "int32", "input", "cpu")
        ir.declare_var(w1, (4,), "int32", "output", "cpu")
        ir.declare_var(w2, (4,), "int32", "output", "cpu")
        y2, y1 = g(x)
        for i in range(4):
            w1[i] = y1[i]
            w2[i] = y2[i]

    func = ir.lower(f, ir.CPU())
    print(func)

    with ir.VarDef([("x", (8,), "int32", "input", "cpu"),
                    ("w1", (4,), "int32", "output", "cpu"),
                    ("w2", (4,), "int32", "output", "cpu"),
                    ("y1", (4,), "int32", "cache", "cpu"),
                    ("y2", (4,), "int32", "cache", "cpu")]) as (x, w1, w2, y1,
                                                                y2):
        with ir.For("i1", 0, 4) as i:
            y1[i] = x[i]
            y2[i] = x[i * 2]
        with ir.For("i2", 0, 4) as i:
            w1[i] = y1[i]
            w2[i] = y2[i]
    std = ir.pop_ast()
    assert std.match(func.body)


def test_func_in_args():

    @ir.inline
    def plus_one(x):
        y = ir.create_var((4,), "int32", "cpu")
        for i in range(4):
            y[i] = x[i] + 1
        return y

    @ir.transform
    def test(x, y):
        ir.declare_var(x, (4,), "int32", "input", "cpu")
        ir.declare_var(y, (4,), "int32", "output", "cpu")
        c = plus_one(plus_one(plus_one(x)))
        for i in range(4):
            y[i] = c[i]

    func = ir.lower(test, ir.CPU())
    print(func)
    code = ir.codegen(func, ir.CPU())
    print(code)
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.array([0, 0, 0, 0], dtype="int32")
    x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
    y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
    ir.Driver(func, code, ir.Device(ir.CPU()))(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([4, 5, 6, 7], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_variadic():

    @ir.inline
    def g(*args, **kvs):
        args[0][0] = 2.0
        args[0][1] = 3.0
        kvs['z'][0] = 4.0
        kvs['z'][1] = 5.0

    @ir.transform
    def f(y, z):
        ir.declare_var(y, (2,), "float32", "output", "cpu")
        ir.declare_var(z, (2,), "float32", "output", "cpu")
        g(y, z=z)

    func = ir.lower(f, ir.CPU())
    print(func)

    with ir.VarDef([("y", (2,), "float32", "output", "cpu"),
                    ("z", (2,), "float32", "output", "cpu")]) as (y, z):
        y[0] = 2.0
        y[1] = 3.0
        z[0] = 4.0
        z[1] = 5.0
    std = ir.pop_ast()
    assert std.match(func.body)


def test_no_deps_on_returned_tensor():
    dev = ir.Device(ir.CPU())

    @ir.inline
    def test_i(a, b):
        'nid: Vc'
        c = ir.create_var((2, 2), "int32", "cpu")
        d = ir.create_var((2, 2), "int32", "cpu")
        for i in range(2):
            for j in range(2):
                b[i, j] = a[i, j]
                c[i, j] = b[i, j] * a[i, j]
                d[i, j] = b[i, j] + a[i, j]
        return c, d

    @ir.transform
    def test(y, c, d):
        ir.declare_var(y, (2, 2), "int32", "output", "cpu")
        ir.declare_var(c, (2, 2), "int32", "output", "cpu")
        ir.declare_var(d, (2, 2), "int32", "output", "cpu")
        cc, dd = test_i(
            ir.capture_var(
                ir.Array(np.array([[1, 2], [3, 4]], dtype=np.int32), dev)), y)
        for i in range(2):
            'nid: Lj'
            'no_deps: cc'
            for j in range(2):
                c[i, j] = cc[i, j]
                d[i, j] = dd[i, j]

    func = ir.lower(test, ir.CPU())
    print(func)

    s = ir.Schedule(func)
    assert s.find('Lj').property.no_deps[0] == s.find('Vc').name
