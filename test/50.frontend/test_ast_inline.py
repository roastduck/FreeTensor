import freetensor as ft


def test_one_arg():

    @ft.transform
    def g(x: ft.Var[(4,), 'int32', 'inout']):
        x[1] = 2

    @ft.transform(verbose=2)
    def f(x: ft.Var[(8, 4), 'int32', 'inout']):
        g(x[3])

    @ft.transform
    def expect(x: ft.Var[(8, 4), 'int32', 'inout']):
        x[3, 1] = 2

    assert expect.body.match(f.body)


def test_one_return():

    @ft.transform
    def g():
        y = ft.empty((4,), "int32")
        y[0] = 1
        y[1] = 3
        y[2] = 2
        y[3] = 4
        return y

    @ft.transform(verbose=2)
    def f(a: ft.Var[(8, 4), 'int32', 'inout']):
        b = g()
        for i in range(4):
            a[2, i] = b[i]

    @ft.transform
    def expect(a: ft.Var[(8, 4), 'int32', 'inout']):
        b = ft.empty((4,), "int32")
        b[0] = 1
        b[1] = 3
        b[2] = 2
        b[3] = 4
        for i in range(4):
            a[2, i] = b[i]

    assert expect.body.match(f.body)


def test_two_returns():

    @ft.transform
    def g():
        y = ft.empty((4,), "int32")
        z = ft.empty((4,), "int32")
        y[0] = 1
        y[1] = 3
        y[2] = 2
        y[3] = 4
        z[0] = -1
        z[1] = -3
        z[2] = -2
        z[3] = -4
        return y, z

    @ft.transform(verbose=2)
    def f(a: ft.Var[(8, 4), 'int32', 'inout']):
        b, c = g()
        for i in range(4):
            a[2, i] = b[i]
            a[3, i] = c[i]

    @ft.transform
    def expect(a: ft.Var[(8, 4), 'int32', 'inout']):
        b = ft.empty((4,), "int32")
        c = ft.empty((4,), "int32")
        b[0] = 1
        b[1] = 3
        b[2] = 2
        b[3] = 4
        c[0] = -1
        c[1] = -3
        c[2] = -2
        c[3] = -4
        for i in range(4):
            a[2, i] = b[i]
            a[3, i] = c[i]

    assert expect.body.match(f.body)


def test_call_as_arg_and_return():

    @ft.transform
    def plus_one(x: ft.Var[(4,), "int32", "input", "cpu"]):
        y = ft.empty((4,), "int32", "cpu")
        for i in range(4):
            y[i] = x[i] + 1
        return y

    @ft.transform(verbose=1)
    def f(x: ft.Var[(4,), "int32", "input", "cpu"]):
        return plus_one(plus_one(plus_one(x)))

    @ft.transform
    def expect(x: ft.Var[(4,), "int32", "input", "cpu"]):
        y1 = ft.empty((4,), "int32", "cpu")
        for i in range(4):
            y1[i] = x[i] + 1
        y2 = ft.empty((4,), "int32", "cpu")
        for i in range(4):
            y2[i] = y1[i] + 1
        y3 = ft.empty((4,), "int32", "cpu")
        for i in range(4):
            y3[i] = y2[i] + 1
        return y3

    assert expect.body.match(f.body)


def test_name_conflict():

    @ft.transform
    def g(x: ft.Var[(), 'int32', 'inout']):
        a = ft.empty((), 'int32')
        a[...] = x[...] + 2
        x[...] = a[...]

    @ft.transform(verbose=1)
    def f(x: ft.Var[(), 'int32', 'inout']):
        a = ft.empty((), 'int32')
        a[...] = x[...] + 1
        g(a)
        x[...] = a[...]

    @ft.transform
    def expect_g(x: ft.Var[(), 'int32', 'inout']):
        b = ft.empty((), 'int32')
        b[...] = x[...] + 2
        x[...] = b[...]

    @ft.transform
    def expect(x: ft.Var[(), 'int32', 'inout']):
        a = ft.empty((), 'int32')
        a[...] = x[...] + 1
        g(a)
        x[...] = a[...]

    assert expect.body.match(f.body)
