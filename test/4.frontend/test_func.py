import ir
import math
import pytest

# TODO: Currently, a callee function must be in the global scope. Can we support a local scope?


@ir.transform
def g_global(y):
    ir.declare_var(y, (2,), "float32", "output", "cpu")
    '''nid: S0'''
    y[0] = 2.0
    '''nid: S1'''
    y[1] = 3.0


@ir.transform
def f_global(y):
    ir.declare_var(y, (2,), "float32", "output", "cpu")
    g_global(y)


def test_basic_call():

    @ir.transform
    def g(y):
        ir.declare_var(y, (2,), "float32", "output", "cpu")
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

    @ir.transform
    def g(y):
        ir.declare_var(y, (2,), "float32", "output", "cpu")
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
    assert len(s.find_all(lambda x: x.nid() == "C1:S0")) == 1
    assert len(s.find_all(lambda x: x.nid() == "C1:S1")) == 1
    assert len(s.find_all(lambda x: x.nid() == "C2:S0")) == 1
    assert len(s.find_all(lambda x: x.nid() == "C2:S1")) == 1


def test_call_with_external_data():
    data = [[0, 1], [2, 3]]

    @ir.transform
    def g(x, y):
        ir.declare_var(x, (2, 2), "int32", "input", "cpu")
        ir.declare_var(y, (2, 2), "int32", "output", "cpu")
        for i in range(2):
            for j in range(2):
                y[i, j] = x[i, j] * 2

    @ir.transform
    def f(y):
        ir.declare_var(y, (2, 2), "int32", "output", "cpu")
        g(data, y)

    func = ir.lower(f, ir.CPU())
    print(func)

    with ir.VarDef("y", (2, 2), "int32", "output", "cpu") as y:
        with ir.VarDef("x", (2, 2), "int32", "cache", "cpu") as x:
            x[0, 0] = 0
            x[0, 1] = 1
            x[1, 0] = 2
            x[1, 1] = 3
            with ir.For("i", 0, 2) as i:
                with ir.For("j", 0, 2) as j:
                    y[i, j] = x[i, j] * 2
    std = ir.pop_ast()
    assert std.match(func.body)


def test_call_with_literal_data():

    @ir.transform
    def g(x, y):
        ir.declare_var(x, (2, 2), "int32", "input", "cpu")
        ir.declare_var(y, (2, 2), "int32", "output", "cpu")
        for i in range(2):
            for j in range(2):
                y[i, j] = x[i, j] * 2

    @ir.transform
    def f(y):
        ir.declare_var(y, (2, 2), "int32", "output", "cpu")
        g([[0, 1], [2, 3]], y)

    func = ir.lower(f, ir.CPU())
    print(func)

    with ir.VarDef("y", (2, 2), "int32", "output", "cpu") as y:
        with ir.VarDef("x", (2, 2), "int32", "cache", "cpu") as x:
            x[0, 0] = 0
            x[0, 1] = 1
            x[1, 0] = 2
            x[1, 1] = 3
            with ir.For("i", 0, 2) as i:
                with ir.For("j", 0, 2) as j:
                    y[i, j] = x[i, j] * 2
    std = ir.pop_ast()
    assert std.match(func.body)


def test_call_with_fixed_dim_at_front():

    @ir.transform
    def g(x1, x2, y):
        ir.declare_var(x1, (4,), "float32", "input", "cpu")
        ir.declare_var(x2, (4,), "float32", "input", "cpu")
        ir.declare_var(y, (4,), "float32", "output", "cpu")
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

    @ir.transform
    def g(x1, x2, y):
        ir.declare_var(x1, (4,), "float32", "input", "cpu")
        ir.declare_var(x2, (4,), "float32", "input", "cpu")
        ir.declare_var(y, (4,), "float32", "output", "cpu")
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

    @ir.transform
    def g(x1, x2, y):
        ir.declare_var(x1, (4,), "float32", "input", "cpu")
        ir.declare_var(x2, (4,), "float32", "input", "cpu")
        ir.declare_var(y, (4,), "float32", "output", "cpu")
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

    @ir.transform
    def g(x1, x2, y):
        ir.declare_var(x1, (), "float32", "input", "cpu")
        ir.declare_var(x2, (), "float32", "input", "cpu")
        ir.declare_var(y, (), "float32", "output", "cpu")
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


def test_external_call():

    @ir.transform
    def func(y):
        ir.declare_var(y, (), "int32", "output", "cpu")
        y[()] = math.gcd(10, 15)

    with ir.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 5
    std = ir.pop_ast()
    assert std.match(func.body)


def test_error_missing_parameters():

    @ir.transform
    def g(y):
        ir.declare_var(y, (2,), "float32", "output", "cpu")
        '''nid: S0'''
        y[0] = 2.0
        '''nid: S1'''
        y[1] = 3.0

    def f(y):
        ir.declare_var(y, (2,), "float32", "output", "cpu")
        g()

    with pytest.raises(ir.InvalidProgram):
        ir.transform(f)
