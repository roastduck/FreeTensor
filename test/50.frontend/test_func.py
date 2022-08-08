import freetensor as ft
from freetensor import debug
import math
import functools
import pytest
import numpy as np


@ft.inline
def g_global(y):
    #! nid: S0
    y[0] = 2.0
    #! nid: S1
    y[1] = 3.0


@ft.transform
def f_global(y):
    y: ft.Var[(2,), "float32", "output", "cpu"]
    g_global(y)


def test_basic_call():

    @ft.inline
    def g(y):
        #! nid: S0
        y[0] = 2.0
        #! nid: S1
        y[1] = 3.0

    @ft.lower(target=ft.CPU(), verbose=1)
    @ft.transform
    def f(y):
        y: ft.Var[(2,), "float32", "output", "cpu"]
        g(y)

    with ft.VarDef("y", (2,), "float32", "output", "cpu") as y:
        y[0] = 2.0
        y[1] = 3.0
    std = ft.pop_ast()
    assert std.match(f.body)


def test_global_functions():

    func = ft.lower(f_global, ft.CPU(), verbose=1)

    with ft.VarDef("y", (2,), "float32", "output", "cpu") as y:
        y[0] = 2.0
        y[1] = 3.0
    std = ft.pop_ast()
    assert std.match(func.body)


def test_called_multiple_times():

    @ft.inline
    def g(y):
        #! nid: S0
        y[0] = 2.0
        #! nid: S1
        y[1] = 3.0

    @ft.lower(target=ft.CPU(), verbose=1)
    @ft.transform
    def f(y1, y2):
        y1: ft.Var[(2,), "float32", "output", "cpu"]
        y2: ft.Var[(2,), "float32", "output", "cpu"]
        #! nid: C1
        g(y1)
        #! nid: C2
        g(y2)

    with ft.VarDef([("y1", (2,), "float32", "output", "cpu"),
                    ("y2", (2,), "float32", "output", "cpu")]) as (y1, y2):
        y1[0] = 2.0
        y1[1] = 3.0
        y2[0] = 2.0
        y2[1] = 3.0
    std = ft.pop_ast()
    assert std.match(f.body)

    s = ft.Schedule(f)
    assert len(s.find_all("C1->S0")) == 1
    assert len(s.find_all("C1->S1")) == 1
    assert len(s.find_all("C2->S0")) == 1
    assert len(s.find_all("C2->S1")) == 1


def test_call_with_external_data():
    data = ft.Array(np.array([[0, 1], [2, 3]], dtype=np.int32))

    @ft.inline
    def g(x, y):
        for i in range(2):
            for j in range(2):
                y[i, j] = x[i, j] * 2

    @ft.lower(target=ft.CPU(), verbose=1)
    @ft.transform
    def f(y):
        y: ft.Var[(2, 2), "int32", "output", "cpu"]
        g(ft.capture_var(data), y)

    with ft.VarDef("y", (2, 2), "int32", "output", "cpu") as y:
        with ft.VarDef("x", (2, 2), "int32", "input", "cpu") as x:
            with ft.For("i", 0, 2) as i:
                with ft.For("j", 0, 2) as j:
                    y[i, j] = x[i, j] * 2
    std = ft.pop_ast()
    assert std.match(f.body)

    code = ft.codegen(f, ft.CPU(), verbose=True)

    y_np = np.zeros((2, 2), dtype="int32")
    y_arr = ft.Array(y_np)
    ft.Driver(f, code, ft.CPU())(y=y_arr)
    y_np = y_arr.numpy()

    assert np.array_equal(y_np, data.numpy() * 2)


def test_call_with_literal_data():
    dev = ft.CPU()

    @ft.inline
    def g(x, y):
        for i in range(2):
            for j in range(2):
                y[i, j] = x[i, j] * 2

    @ft.lower(target=ft.CPU(), verbose=1)
    @ft.transform
    def f(y):
        y: ft.Var[(2, 2), "int32", "output", "cpu"]
        g(ft.capture_var(ft.Array(np.array([[0, 1], [2, 3]], dtype=np.int32))),
          y)

    with ft.VarDef("y", (2, 2), "int32", "output", "cpu") as y:
        with ft.VarDef("x", (2, 2), "int32", "input", "cpu") as x:
            with ft.For("i", 0, 2) as i:
                with ft.For("j", 0, 2) as j:
                    y[i, j] = x[i, j] * 2
    std = ft.pop_ast()
    assert std.match(f.body)

    code = ft.codegen(f, ft.CPU(), verbose=True)

    y_np = np.zeros((2, 2), dtype="int32")
    y_arr = ft.Array(y_np)
    ft.Driver(f, code, dev)(y=y_arr)
    y_np = y_arr.numpy()

    assert np.array_equal(y_np, np.array([[0, 1], [2, 3]], dtype=np.int32) * 2)


def test_call_with_fixed_dim_at_front():

    @ft.inline
    def g(x1, x2, y):
        for i in range(4):
            y[i] = x1[i] + x2[i]

    @ft.lower(target=ft.CPU(), verbose=1)
    @ft.transform
    def f(x1, x2, y):
        x1: ft.Var[(4, 4), "float32", "input", "cpu"]
        x2: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        for i in range(4):
            g(x1[i], x2[i], y[i])

    with ft.VarDef([("x1", (4, 4), "float32", "input", "cpu"),
                    ("x2", (4, 4), "float32", "input", "cpu"),
                    ("y", (4, 4), "float32", "output", "cpu")]) as (x1, x2, y):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 4) as j:
                y[i, j] = x1[i, j] + x2[i, j]
    std = ft.pop_ast()
    assert std.match(f.body)


def test_call_with_fixed_dim_at_back():

    @ft.inline
    def g(x1, x2, y):
        for i in range(4):
            y[i] = x1[i] + x2[i]

    @ft.lower(target=ft.CPU(), verbose=1)
    @ft.transform
    def f(x1, x2, y):
        x1: ft.Var[(4, 4), "float32", "input", "cpu"]
        x2: ft.Var[(4, 4), "float32", "input", "cpu"]
        y: ft.Var[(4, 4), "float32", "output", "cpu"]
        for i in range(4):
            g(x1[:, i], x2[:, i], y[:, i])

    with ft.VarDef([("x1", (4, 4), "float32", "input", "cpu"),
                    ("x2", (4, 4), "float32", "input", "cpu"),
                    ("y", (4, 4), "float32", "output", "cpu")]) as (x1, x2, y):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 4) as j:
                y[j, i] = x1[j, i] + x2[j, i]
    std = ft.pop_ast()
    assert std.match(f.body)


def test_call_with_slice():

    @ft.inline
    def g(x1, x2, y):
        for i in range(4):
            y[i] = x1[i] + x2[i]

    @ft.lower(target=ft.CPU(), verbose=1)
    @ft.transform
    def f(x1, x2, y):
        x1: ft.Var[(5,), "float32", "input", "cpu"]
        x2: ft.Var[(5,), "float32", "input", "cpu"]
        y: ft.Var[(5,), "float32", "output", "cpu"]
        y[0] = 0.
        g(x1[1:], x2[1:], y[1:])

    with ft.VarDef([("x1", (5,), "float32", "input", "cpu"),
                    ("x2", (5,), "float32", "input", "cpu"),
                    ("y", (5,), "float32", "output", "cpu")]) as (x1, x2, y):
        y[0] = 0.
        with ft.For("i", 0, 4) as i:
            y[i + 1] = x1[i + 1] + x2[i + 1]
    std = ft.pop_ast()
    assert std.match(f.body)


def test_call_with_scalar():

    @ft.inline
    def g(x1, x2, y):
        y[()] = x1[()] + x2[()]

    @ft.lower(target=ft.CPU(), verbose=1)
    @ft.transform
    def f(x1, x2, y):
        x1: ft.Var[(4,), "float32", "input", "cpu"]
        x2: ft.Var[(4,), "float32", "input", "cpu"]
        y: ft.Var[(4,), "float32", "output", "cpu"]
        for i in range(4):
            g(x1[i], x2[i], y[i])

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x1, x2, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x1[i] + x2[i]
    std = ft.pop_ast()
    assert std.match(f.body)


def test_call_with_literal_scalar():

    @ft.inline
    def g(x1, x2, y):
        y[()] = x1 + x2

    @ft.lower(target=ft.CPU(), verbose=1)
    @ft.transform
    def f(y):
        y: ft.Var[(4,), "float32", "output", "cpu"]
        for i in range(4):
            g(1, 2, y[i])

    with ft.VarDef("y", (4,), "float32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = 3
    std = ft.pop_ast()
    assert std.match(f.body)


def test_external_call():

    @ft.transform
    def func(y):
        y: ft.Var[(), "int32", "output", "cpu"]
        y[()] = math.gcd(10, 15)

    with ft.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 5
    std = ft.pop_ast()
    assert std.match(func.body)


def test_use_external_call_to_build_runtime_ops():

    @ft.inline
    def g(x1, x2, x3, y):
        y[()] = functools.reduce(lambda x, y: x * y, [x1, x2, x3])

    @ft.lower(target=ft.CPU(), verbose=1)
    @ft.transform
    def f(x1, x2, x3, y):
        x1: ft.Var[(), "int32", "input", "cpu"]
        x2: ft.Var[(), "int32", "input", "cpu"]
        x3: ft.Var[(), "int32", "input", "cpu"]
        y: ft.Var[(), "int32", "output", "cpu"]
        g(x1, x2, x3, y)

    with ft.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("x3", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x1, x2, x3, y):
        y[()] = x1[()] * x2[()] * x3[()]
    std = ft.pop_ast()
    assert std.match(f.body)


def test_error_missing_parameters():

    @ft.inline
    def g(y):
        #! nid: S0
        y[0] = 2.0
        #! nid: S1
        y[1] = 3.0

    def f(y):
        y: ft.Var[(2,), "float32", "output", "cpu"]
        g()

    with pytest.raises(ft.StagingError):
        ft.transform(f)


def test_return():
    dev = ft.CPU()

    @ft.inline
    def test_i(a, b):
        c = ft.empty((2, 2), "int32", "cpu")
        d = ft.empty((2, 2), "int32", "cpu")
        for i in range(2):
            for j in range(2):
                b[i, j] = a[i, j]
                c[i, j] = b[i, j] * a[i, j]
                d[i, j] = b[i, j] + a[i, j]
        return c, d

    @ft.lower(target=ft.CPU(), skip_passes=['prop_one_time_use'], verbose=1)
    @ft.transform
    def test(y, c, d):
        y: ft.Var[(2, 2), "int32", "output", "cpu"]
        c: ft.Var[(2, 2), "int32", "output", "cpu"]
        d: ft.Var[(2, 2), "int32", "output", "cpu"]
        c1, d1 = test_i(
            ft.capture_var(ft.Array(np.array([[1, 2], [3, 4]],
                                             dtype=np.int32))), y)
        for i in range(2):
            for j in range(2):
                c[i, j] = c1[i, j]
                d[i, j] = d1[i, j]

    with ft.VarDef([("y", (2, 2), "int32", "output", "cpu"),
                    ("c", (2, 2), "int32", "output", "cpu"),
                    ("d", (2, 2), "int32", "output", "cpu")]) as (y, c, d):
        with ft.VarDef("a", (2, 2), "int32", "input", "cpu") as a:
            with ft.VarDef([("c1", (2, 2), "int32", "cache", "cpu"),
                            ("d1", (2, 2), "int32", "cache", "cpu")]) as (c1,
                                                                          d1):
                with ft.For("i", 0, 2) as i:
                    with ft.For("j", 0, 2) as j:
                        y[i, j] = a[i, j]
                        c1[i, j] = y[i, j] * a[i, j]
                        d1[i, j] = y[i, j] + a[i, j]
                with ft.For("i1", 0, 2) as i:
                    with ft.For("j1", 0, 2) as j:
                        c[i, j] = c1[i, j]
                        d[i, j] = d1[i, j]
    std = ft.pop_ast(verbose=True)
    assert std.match(test.body)


def test_return_returned_value():

    @ft.inline
    def h(x):
        y1 = ft.empty((4,), "int32", "cpu")
        y2 = ft.empty((4,), "int32", "cpu")
        for i in range(4):
            y1[i] = x[i]
            y2[i] = x[i * 2]
        return y1, y2

    @ft.inline
    def g(x):
        y1, y2 = h(x)
        return y2, y1

    @ft.lower(target=ft.CPU(), skip_passes=['prop_one_time_use'], verbose=1)
    @ft.transform
    def f(x, w1, w2):
        x: ft.Var[(8,), "int32", "input", "cpu"]
        w1: ft.Var[(4,), "int32", "output", "cpu"]
        w2: ft.Var[(4,), "int32", "output", "cpu"]
        y2, y1 = g(x)
        for i in range(4):
            w1[i] = y1[i]
            w2[i] = y2[i]

    with ft.VarDef([("x", (8,), "int32", "input", "cpu"),
                    ("w1", (4,), "int32", "output", "cpu"),
                    ("w2", (4,), "int32", "output", "cpu"),
                    ("y1", (4,), "int32", "cache", "cpu"),
                    ("y2", (4,), "int32", "cache", "cpu")]) as (x, w1, w2, y1,
                                                                y2):
        with ft.For("i1", 0, 4) as i:
            y1[i] = x[i]
            y2[i] = x[i * 2]
        with ft.For("i2", 0, 4) as i:
            w1[i] = y1[i]
            w2[i] = y2[i]
    std = ft.pop_ast()
    assert std.match(f.body)


def test_func_in_args():

    @ft.inline
    def plus_one(x):
        y = ft.empty((4,), "int32", "cpu")
        for i in range(4):
            y[i] = x[i] + 1
        return y

    @ft.build_binary(device=ft.CPU())
    @ft.codegen(target=ft.CPU(), verbose=True)
    @ft.lower(target=ft.CPU(), verbose=1)
    @ft.transform
    def f(x, y):
        x: ft.Var[(4,), "int32", "input", "cpu"]
        y: ft.Var[(4,), "int32", "output", "cpu"]
        c = plus_one(plus_one(plus_one(x)))
        for i in range(4):
            y[i] = c[i]

    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.array([0, 0, 0, 0], dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    f(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([4, 5, 6, 7], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_variadic():

    @ft.inline
    def g(*args, **kvs):
        args[0][0] = 2.0
        args[0][1] = 3.0
        kvs['z'][0] = 4.0
        kvs['z'][1] = 5.0

    @ft.lower(target=ft.CPU(), verbose=1)
    @ft.transform
    def f(y, z):
        y: ft.Var[(2,), "float32", "output", "cpu"]
        z: ft.Var[(2,), "float32", "output", "cpu"]
        g(y, z=z)

    with ft.VarDef([("y", (2,), "float32", "output", "cpu"),
                    ("z", (2,), "float32", "output", "cpu")]) as (y, z):
        y[0] = 2.0
        y[1] = 3.0
        z[0] = 4.0
        z[1] = 5.0
    std = ft.pop_ast()
    assert std.match(f.body)


def test_no_deps_on_returned_tensor():
    dev = ft.CPU()

    @ft.inline
    def test_i(a, b):
        #! nid: Vc
        c = ft.empty((2, 2), "int32", "cpu")
        d = ft.empty((2, 2), "int32", "cpu")
        for i in range(2):
            for j in range(2):
                b[i, j] = a[i, j]
                c[i, j] = b[i, j] * a[i, j]
                d[i, j] = b[i, j] + a[i, j]
        return c, d

    @ft.lower(target=ft.CPU(), skip_passes=['prop_one_time_use'], verbose=1)
    @ft.transform
    def test(y, c, d):
        y: ft.Var[(2, 2), "int32", "output", "cpu"]
        c: ft.Var[(2, 2), "int32", "output", "cpu"]
        d: ft.Var[(2, 2), "int32", "output", "cpu"]
        cc, dd = test_i(
            ft.capture_var(ft.Array(np.array([[1, 2], [3, 4]],
                                             dtype=np.int32))), y)
        for i in range(2):
            #! nid: Lj
            #! no_deps: cc
            for j in range(2):
                c[i, j] = cc[i, j]
                d[i, j] = dd[i, j]

    s = ft.Schedule(test)
    assert s.find('Lj').property.no_deps[0] == s.find('Vc').name


def test_late_definition():

    @ft.inline
    def caller(x):
        callee(x)

    @ft.inline
    def callee(x):
        x[()] = x[()] + 1

    @ft.transform
    def test(x: ft.Var[(), "int32", "inout"]):
        caller(x)

    @ft.transform
    def test_expected(x: ft.Var[(), "int32", "inout"]):
        x[()] = x[()] + 1

    assert test.body.match(test_expected.body)


@ft.inline
def caller_global(x):
    callee_global(x)


@ft.inline
def callee_global(x):
    x[()] = x[()] + 1


def test_late_definition_global():

    @ft.transform
    def test(x: ft.Var[(), "int32", "inout"]):
        caller_global(x)

    @ft.transform
    def test_expected(x: ft.Var[(), "int32", "inout"]):
        x[()] = x[()] + 1

    assert test.body.match(test_expected.body)
