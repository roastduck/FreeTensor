import freetensor as ft


def test_basic():
    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.If(i < 2):
                y1[i] = 0
            with ft.Else():
                y1[i] = 1
            with ft.If(i < 2):
                y2[i] = 2
            with ft.Else():
                y2[i] = 3
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.separate_tail()
    ast = s.ast()
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 2) as i:
            y1[i] = 0
            y2[i] = 2
        with ft.For("i", 2, 4) as i:
            y1[i] = 1
            y2[i] = 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_multiple_cond():
    with ft.VarDef([("y1", (5,), "int32", "output", "cpu"),
                    ("y2", (5,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 5) as i:
            with ft.If(i < 2):
                y1[i] = 0
            with ft.Else():
                y1[i] = 1
            with ft.If(i < 3):
                y2[i] = 2
            with ft.Else():
                y2[i] = 3
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.separate_tail()
    ast = s.ast()
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("y1", (5,), "int32", "output", "cpu"),
                    ("y2", (5,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 2) as i:
            y1[i] = 0
            y2[i] = 2
        y1[2] = 1
        y2[2] = 2
        with ft.For("i", 3, 5) as i:
            y1[i] = 1
            y2[i] = 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_eq():
    with ft.VarDef("y", (5,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 5) as i:
            with ft.If(i == 2):
                y[i] = 0
            with ft.Else():
                y[i] = 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.separate_tail()
    ast = s.ast()
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("y", (5,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 2) as i:
            y[i] = 1
        y[2] = 0
        with ft.For("i", 3, 5) as i:
            y[i] = 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_tiled():
    with ft.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 3) as i:
            with ft.For("j", 0, 4) as j:
                with ft.If(4 * i + j < 10):
                    y[4 * i + j] = 4 * i + j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.separate_tail()
    ast = s.ast()
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 2) as i:
            with ft.For("j", 0, 4) as j:
                y[4 * i + j] = 4 * i + j
        with ft.For("j", 0, 2) as j:
            y[8 + j] = 8 + j
    std = ft.pop_ast()

    assert std.match(ast)


def test_dynamic_tiled():
    with ft.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ft.Assert(n[()] > 0):
            with ft.VarDef("y", (n[()],), "int32", "output", "cpu") as y:
                with ft.For("i", 0, ft.ceildiv(n[()], 4)) as i:
                    with ft.For("j", 0, 4) as j:
                        with ft.If(4 * i + j < n[()]):
                            y[4 * i + j] = 4 * i + j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.separate_tail()
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ft.Assert(n[()] > 0):
            with ft.VarDef("y", (n[()],), "int32", "output", "cpu") as y:
                with ft.For("i", 0, ft.any()) as i:
                    with ft.For("j", 0, 4) as j:
                        y[4 * i + j] = 4 * i + j
                with ft.For("j", 0, ft.any()) as j:
                    y[ft.any()] = ft.any()
    std = ft.pop_ast()

    assert std.match(ast)


def test_1d_stencil():
    with ft.VarDef([("x", (10,), "int32", "input", "cpu"),
                    ("y", (10,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 10) as i:
            y[i] = x[i]
            with ft.If(i - 1 >= 0):
                y[i] = y[i] + x[i - 1]
            with ft.If(i + 1 < 10):
                y[i] = y[i] + x[i + 1]
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.separate_tail()
    ast = s.ast()
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (10,), "int32", "input", "cpu"),
                    ("y", (10,), "int32", "output", "cpu")]) as (x, y):
        y[0] = x[0] + x[1]
        with ft.For("i", 1, 9) as i:
            y[i] = ft.any()
        y[9] = x[9] + x[8]
    std = ft.pop_ast()

    assert std.match(ast)


def test_duplicate_vardef():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[()] = x[i]
                with ft.If(t[()] < 0):
                    t[()] *= -1
                with ft.If(i < 2):
                    y[i] = t[()] + 1
                with ft.Else():
                    y[i] = t[()] + 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.separate_tail()
    ast = s.ast()
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 2) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[()] = x[i]
                with ft.If(t[()] < 0):
                    t[()] *= -1
                y[i] = t[()] + 1
        with ft.For("i", 2, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[()] = x[i]
                with ft.If(t[()] < 0):
                    t[()] *= -1
                y[i] = t[()] + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_no_duplicate_vardef():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[()] = x[i]
                with ft.If(t[()] < 0):
                    t[()] *= -1
                with ft.If(i < 2):
                    y[i] = t[()] + 1
                with ft.Else():
                    y[i] = t[()] + 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.separate_tail(True)
    ast = s.ast()
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[()] = x[i]
                with ft.If(t[()] < 0):
                    t[()] *= -1
                with ft.If(i < 2):
                    y[i] = t[()] + 1
                with ft.Else():
                    y[i] = t[()] + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_no_hoisting_loop_variant():

    @ft.transform
    def foo(x):
        x: ft.Var[(32,), 'int32', 'output', 'cpu']
        a = ft.empty((), 'int32', 'cpu')
        a[()] = 10
        for i in range(32):
            a[()] = 20
            if i < a[()]:
                x[i] = i
            else:
                x[i] = 32 - i

    s = ft.Schedule(foo)
    s.separate_tail()
    print(s.func())
