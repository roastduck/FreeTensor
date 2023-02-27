from dataclasses import dataclass
from typing import Callable

import pytest
import freetensor as ft
import numpy as np


def test_hello_world():

    def test(x):
        x: ft.Var[(4, 4), "float32", "output"]
        x[2, 3] = 2.0
        x[1, 0] = 3.0

    func = ft.lower(ft.transform(test), verbose=1)
    code = ft.codegen(func)

    x_np = np.zeros((4, 4), dtype="float32")
    x_arr = ft.Array(x_np)
    ft.build_binary(code)(x=x_arr)
    x_np = x_arr.numpy()

    x_std = np.zeros((4, 4), dtype="float32")
    x_std[2, 3] = 2.0
    x_std[1, 0] = 3.0
    x_func = np.zeros((4, 4), dtype="float32")
    test(x_func)
    assert np.array_equal(x_np, x_std)
    assert np.array_equal(x_func, x_std)


def test_declare_var_in_function_declaration():

    def test(x: ft.Var[(4, 4), "float32", "output"]):
        x[2, 3] = 2.0
        x[1, 0] = 3.0

    func = ft.lower(ft.transform(test), verbose=1)
    code = ft.codegen(func)

    x_np = np.zeros((4, 4), dtype="float32")
    x_arr = ft.Array(x_np)
    ft.build_binary(code)(x=x_arr)
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
        x: ft.Var[(), "int32", "input"]
        y: ft.Var[(), "int32", "output"]
        y[()] = x[()] * 2 + 1

    func = ft.lower(ft.transform(test))
    code = ft.codegen(func)
    x_np = np.array(5, dtype="int32")
    y_np = np.array(0, dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.array(0, dtype="int32")
    test(x_np, y_func)

    assert y_np[()] == 11
    assert y_func[()] == 11


def test_return_value_and_runtime_allocation():

    @ft.optimize(verbose=1)
    def test(x):
        x: ft.Var[(), "int32"]
        y = ft.empty((), "int32")
        y[()] = x[()] * 2 + 1
        return y

    y_arr = test(np.array(5, dtype="int32"))
    y_np = y_arr.numpy()

    assert y_np[()] == 11


def test_multiple_return_values():

    @ft.optimize(verbose=1)
    def test(x):
        x: ft.Var[(), "int32"]
        y = ft.empty((), "int32")
        z = ft.empty((), "int32")
        y[()] = x[()] * 2
        z[()] = x[()] * 2 + 1
        return y, z

    y_arr, z_arr = test(np.array(5, dtype="int32"))
    y_np = y_arr.numpy()
    z_np = z_arr.numpy()

    assert y_np[()] == 10
    assert z_np[()] == 11


def test_named_return_values():

    @ft.optimize(verbose=1)
    def test(x):
        x: ft.Var[(), "int32"]
        y = ft.empty((), "int32")
        z = ft.empty((), "int32")
        y[()] = x[()] * 2
        z[()] = x[()] * 2 + 1
        return y, z

    ret = test(np.array(5, dtype="int32"))
    y_np = ret['y'].numpy()
    z_np = ret['z'].numpy()

    assert y_np[()] == 10
    assert z_np[()] == 11


def test_redundant_parameter():

    @ft.optimize(verbose=1)
    def func(x: ft.Var[(), "int32"], y: ft.Var[(), "int32"]):
        z = ft.empty((), "int32")
        z[...] = x[...]  # `y` is unused
        return z

    z_arr = func(np.array(2, dtype="int32"), np.array(3, dtype="int32"))
    assert z_arr.numpy().item() == 2


def test_for():

    def test(x, y):
        x: ft.Var[(4,), "int32", "input"]
        y: ft.Var[(4,), "int32", "output"]
        for i in range(0, 4):
            y[i] = x[i] + 1

    func = ft.lower(ft.transform(test))
    code = ft.codegen(func)
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((4,), dtype="int32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.zeros((4,), dtype="int32")
    test(x_np, y_func)

    y_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(y_func, y_std)


def test_if():

    def test(y):
        y: ft.Var[(4,), "int32", "output"]
        for i in range(0, 4):
            if i < 2:
                y[i] = 0
            else:
                y[i] = 1

    func = ft.lower(ft.transform(test))
    code = ft.codegen(func)
    y_np = np.zeros((4,), dtype="int32")
    y_arr = ft.Array(y_np)
    ft.build_binary(code)(y=y_arr)
    y_np = y_arr.numpy()
    y_func = np.zeros((4,), dtype="int32")
    test(y_func)

    y_std = np.array([0, 0, 1, 1], dtype="int32")
    assert np.array_equal(y_np, y_std)
    assert np.array_equal(y_func, y_std)


def test_static_if():

    flag = True

    def test(y):
        y: ft.Var[(4,), "int32", "output"]
        for i in range(0, 4):
            value = 0
            if flag:
                value = 1
            y[i] = value

    func = ft.lower(ft.transform(test))
    code = ft.codegen(func)
    y_np = np.zeros((4,), dtype="int32")
    y_arr = ft.Array(y_np)
    ft.build_binary(code)(y=y_arr)
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
        y: ft.Var[(4,), "int32", "output"]
        f(y, 0)

    func = ft.lower(test)
    code = ft.codegen(func)
    y_np = np.zeros((4,), dtype="int32")
    y_arr = ft.Array(y_np)
    ft.build_binary(code)(y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.array([1, 1, 1, 1], dtype="int32")
    assert np.array_equal(y_np, y_std)


def test_for_range():

    def test(x):
        x: ft.Var[(4,), "int32", "output"]
        for i in range(4):
            x[i] += 1

    func = ft.lower(ft.transform(test))
    code = ft.codegen(func)
    x_np = np.array([1, 2, 3, 4], dtype="int32")
    x_arr = ft.Array(x_np)
    ft.build_binary(code)(x=x_arr)
    x_np = x_arr.numpy()
    x_func = np.array([1, 2, 3, 4], dtype="int32")
    test(x_func)

    x_std = np.array([2, 3, 4, 5], dtype="int32")
    assert np.array_equal(x_np, x_std)
    assert np.array_equal(x_func, x_std)


def test_std_func_alias():

    def test(x):
        x: ft.Var[(4, 4), "float32", "output"]
        x[2, 3] = 2.0
        x[1, 0] = 3.0

    func = ft.lower(ft.transform(test), verbose=1)
    code = ft.codegen(func)

    x_np = np.zeros((4, 4), dtype="float32")
    x_arr = ft.Array(x_np)
    ft.build_binary(code)(x=x_arr)
    x_np = x_arr.numpy()

    x_std = np.zeros((4, 4), dtype="float32")
    x_std[2, 3] = 2.0
    x_std[1, 0] = 3.0
    x_func = np.zeros((4, 4), dtype="float32")
    test(x_func)
    assert np.array_equal(x_np, x_std)
    assert np.array_equal(x_func, x_std)


def test_assert():

    @ft.transform(verbose=1)
    def test(x1, x2, y1, y2):
        x1: ft.Var[(4,), "int32", "input"]
        x2: ft.Var[(4,), "int32", "input"]
        y1: ft.Var[(4,), "int32", "output"]
        y2: ft.Var[(4,), "int32", "output"]
        for i in range(4):
            y1[i] = x1[i] + x2[i]
            assert x1[i] < x2[i]
            y2[i] = ft.min(x1[i], x2[i])

    with ft.VarDef([("x1", (4,), "int32", "input"),
                    ("x2", (4,), "int32", "input"),
                    ("y1", (4,), "int32", "output"),
                    ("y2", (4,), "int32", "output")]) as (x1, x2, y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = x1[i] + x2[i]
            with ft.Assert(x1[i] < x2[i]):
                y2[i] = ft.min(x1[i], x2[i])
    std = ft.pop_ast()

    assert std.match(test.body)


def test_immediate_var_return():

    @ft.optimize(verbose=1)
    def test(x: ft.Var[(), "int32"]):
        return ft.var([0, 1, x[()]], "int32")

    y_arr = test(np.array(2, dtype="int32"))
    y_np = y_arr.numpy()

    assert np.all(y_np == np.array([0, 1, 2]))


def test_directly_returning_argument():

    @ft.optimize(verbose=1)
    def test(x: ft.Var[(3,), "int32"]):
        return x

    y_arr = test(np.array([0, 1, 2], dtype="int32"))
    y_np = y_arr.numpy()

    assert np.all(y_np == np.array([0, 1, 2]))


def test_modify_and_return_argument():

    @ft.optimize(verbose=1)
    def test(x: ft.Var[(3,), "int32", "inout"]):
        for i in range(3):
            x[i] += 1
        return x

    y_arr = test(np.array([0, 1, 2], dtype="int32"))
    y_np = y_arr.numpy()

    assert np.all(y_np == np.array([1, 2, 3]))


def test_inline_annotation():

    def test(x: ft.Var[(4, 4), "float32", "output"]):
        x[2, 3] = 2.0
        x[1, 0] = 3.0

    func = ft.lower(ft.transform(test), verbose=1)
    code = ft.codegen(func)

    x_np = np.zeros((4, 4), dtype="float32")
    x_arr = ft.Array(x_np)
    ft.build_binary(code)(x=x_arr)
    x_np = x_arr.numpy()

    x_std = np.zeros((4, 4), dtype="float32")
    x_std[2, 3] = 2.0
    x_std[1, 0] = 3.0
    x_func = np.zeros((4, 4), dtype="float32")
    test(x_func)
    assert np.array_equal(x_np, x_std)
    assert np.array_equal(x_func, x_std)


def test_while():

    @dataclass
    class WhilePredicate(ft.StagedPredicate):
        inner: ft.StagedPredicate

        def logical_and(self, other: ft.StagedPredicate):
            return self.inner.logical_and(other)

        def logical_or(self, other: ft.StagedPredicate):
            return self.inner.logical_or(other)

        def logical_not(self):
            return self.inner.logical_not()

        def if_then_else_stmt(self, then_stmt, else_stmt):
            self.inner.if_then_else_stmt(then_stmt, else_stmt)

        def if_then_else_expr(self, then_expr, else_expr):
            return self.inner.if_then_else_expr(then_expr, else_expr)

        @ft.inline
        def while_stmt(self, body: Callable[[], None]):
            for _ in range(2**31 - 1):
                if self.inner:
                    body()

        def assert_stmt(self):
            self.inner.assert_stmt()

    @ft.transform
    def test(x: ft.Var[(), "float32", "output"]):
        while WhilePredicate(x[()] > 0):
            x[()] -= 1

    @ft.transform
    def test_expected(x: ft.Var[(), "float32", "output"]):
        for _ in range(2**31 - 1):
            if x[()] > 0:
                x[()] -= 1

    assert test.body.match(test_expected.body)


def test_return_in_static_if():
    do_add = True

    @ft.transform
    def test(x: ft.Var[(), 'float32']):
        if do_add:
            y = ft.empty((), 'float32')
            y[()] = x[()] + 1
            return y
        else:
            return x

    @ft.transform
    def test_expected(x: ft.Var[(), 'float32']):
        y = ft.empty((), 'float32')
        y[()] = x[()] + 1
        return y

    assert test.body.match(test_expected.body)


def test_error_return_in_dynamic_if():
    with pytest.raises(ft.StagingError):

        @ft.transform
        def test(x: ft.Var[(), 'float32'], do_add: ft.Var[(), 'int32']):
            if do_add == 1:
                y = ft.empty((), 'float32')
                y[()] = x[()] + 1
                return y
            else:
                return x


def test_break_from_static_loop():

    @ft.transform
    def test(x: ft.Var[(), 'float32', 'inout']):
        for i in ft.static_range(5):
            if i == 3:
                break
            x[()] += 1

    @ft.transform
    def test_expected(x: ft.Var[(), 'float32', 'inout']):
        x[()] += 1
        x[()] += 1
        x[()] += 1

    assert test.body.match(test_expected.body)


def test_continue_from_static_loop():

    @ft.transform
    def test(x: ft.Var[(), 'float32', 'inout']):
        for i in ft.static_range(5):
            if i == 3:
                continue
            x[()] += i

    @ft.transform
    def test_expected(x: ft.Var[(), 'float32', 'inout']):
        x[()] += 0
        x[()] += 1
        x[()] += 2
        x[()] += 4

    assert test.body.match(test_expected.body)


def test_break_from_dynamic_loop():
    with pytest.raises(ft.StagingError):

        @ft.transform
        def test(x: ft.Var[(), 'float32', 'inout']):
            for i in ft.dynamic_range(5):
                if i == 3:
                    break
                x[()] += 1


def test_continue_from_dynamic_loop():
    with pytest.raises(ft.StagingError):

        @ft.transform
        def test(x: ft.Var[(), 'float32', 'inout']):
            for i in ft.dynamic_range(5):
                if i == 3:
                    continue
                x[()] += i


def test_tuple_for():

    @ft.transform(verbose=2)
    def test(x: ft.Var[(), 'float32']):
        y = ft.empty((), 'float32')
        for u, (a, b) in [(x[()], (1, 2)), (x[()], (3, 4))]:
            print(u, a, b)
            y[()] += a * u + b
        return y

    @ft.transform
    def test_expected(x: ft.Var[(), 'float32']):
        y = ft.empty((), 'float32')
        y[()] += 1 * x[()] + 2
        y[()] += 3 * x[()] + 4
        return y

    assert test.body.match(test_expected.body)


@dataclass
class DummyAssigned:
    attr = None


def test_attribute_assign():

    @ft.transform
    def test():
        t = DummyAssigned()
        t.attr = ft.var(1, 'float32')
        return t.attr

    @ft.transform
    def test_expected():
        attr = ft.var(1, 'float32')
        return attr

    assert test.body.match(test_expected.body)


def test_not_captured_annotation():
    ft_alter = ft

    @ft.transform()
    def test(x: ft_alter.Var[(4,), "float32"]):
        y = x + 1
        return y

    @ft.transform()
    def test_expected(x: ft.Var[(4,), "float32"]):
        y = x + 1
        return y

    assert test.body.match(test_expected.body)
