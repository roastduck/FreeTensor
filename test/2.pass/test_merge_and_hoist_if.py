import freetensor as ft


def test_merge():
    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.If(x[i] < 2):
                y1[i] = 0
            with ft.Else():
                y1[i] = 1
            with ft.If(x[i] < 2):
                y2[i] = 2
            with ft.Else():
                y2[i] = 3
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.If(x[i] < 2):
                y1[i] = 0
                y2[i] = 2
            with ft.Else():
                y1[i] = 1
                y2[i] = 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_no_merge_different_cond():
    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (5,), "int32", "output", "cpu"),
        ("y2", (5,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 5) as i:
            with ft.If(x[i] < 2):
                y1[i] = 0
            with ft.Else():
                y1[i] = 1
            with ft.If(x[i] < 3):
                y2[i] = 2
            with ft.Else():
                y2[i] = 3
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (5,), "int32", "output", "cpu"),
        ("y2", (5,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 5) as i:
            with ft.If(x[i] < 2):
                y1[i] = 0
            with ft.Else():
                y1[i] = 1
            with ft.If(x[i] < 3):
                y2[i] = 2
            with ft.Else():
                y2[i] = 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_no_merge_may_update():
    with ft.VarDef("a", (4,), "int32", "inout", "cpu") as a:
        with ft.For("i", 0, 4) as i:
            with ft.If(a[i] > 10):
                a[i] = a[i] / 2
            with ft.If(a[i] > 10):
                a[i] = a[i] / 2
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef("a", (4,), "int32", "inout", "cpu") as a:
        with ft.For("i", 0, 4) as i:
            with ft.If(a[i] > 10):
                a[i] = a[i] / 2
            with ft.If(a[i] > 10):
                a[i] = a[i] / 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_hoist():
    with ft.VarDef([("x", (3, 4), "int32", "input", "cpu"),
                    ("y", (3, 4), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 3) as i:
            with ft.For("j", 0, 4) as j:
                with ft.If(i % 2 == 0):
                    y[i, j] = x[i, j]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=['use_builtin_div'])
    print(ast)

    with ft.VarDef([("x", (3, 4), "int32", "input", "cpu"),
                    ("y", (3, 4), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 3) as i:
            with ft.If(i % 2 == 0):
                with ft.For("j", 0, 4) as j:
                    y[i, j] = x[i, j]
    std = ft.pop_ast()

    assert std.match(ast)


def test_not_hoisting_not_pure_nested():
    with ft.VarDef([
        ("x", (3, 4), "int32", "input", "cpu"),
        ("y", (3, 4), "int32", "output", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 3) as i:
            with ft.For("j", 0, 4) as j:
                y[i, j] = 0
                with ft.If(i % 2 == 0):
                    y[i, j] = x[i, j]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=['use_builtin_div'])
    print(ast)

    with ft.VarDef([
        ("x", (3, 4), "int32", "input", "cpu"),
        ("y", (3, 4), "int32", "output", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 3) as i:
            with ft.For("j", 0, 4) as j:
                y[i, j] = 0
                with ft.If(i % 2 == 0):
                    y[i, j] = x[i, j]
    std = ft.pop_ast()

    assert std.match(ast)


def test_not_hoisting_when_being_updated():
    with ft.VarDef([
        ("n", (), "int32", "inout", "cpu"),
        ("x", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "inout", "cpu"),
    ]) as (n, x, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(n[()] < x[i]):
                y[i] = 0
                n[()] = n[()] + x[i]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("n", (), "int32", "inout", "cpu"),
        ("x", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "inout", "cpu"),
    ]) as (n, x, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(n[()] < x[i]):
                y[i] = 0
                n[()] = n[()] + x[i]
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_hoist_then_merge():
    with ft.VarDef([("x", (3, 4), "int32", "input", "cpu"),
                    ("y", (3, 4), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 3) as i:
            with ft.For("j", 0, 4) as j:
                with ft.If(i % 2 == 0):
                    y[i, j] = x[i, j]
            with ft.If(i % 2 == 0):
                y[i, 0] = y[i, 0] + 1
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=['use_builtin_div'])
    print(ast)

    with ft.VarDef([("x", (3, 4), "int32", "input", "cpu"),
                    ("y", (3, 4), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 3) as i:
            with ft.If(i % 2 == 0):
                with ft.For("j", 0, 4) as j:
                    y[i, j] = x[i, j]
                y[i, 0] = y[i, 0] + 1
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_merge_then_hoist():
    with ft.VarDef([
        ("x", (), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "inout", "cpu"),
        ("y2", (4,), "int32", "inout", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.If(x[()] < 2):
                y1[i] = 0
            with ft.If(x[()] < 2):
                y2[i] = 2
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef("x", (), "int32", "input", "cpu") as x:
        with ft.If(x[()] < 2):
            with ft.VarDef([
                ("y1", (4,), "int32", "inout", "cpu"),
                ("y2", (4,), "int32", "inout", "cpu"),
            ]) as (y1, y2):
                with ft.For("i", 0, 4) as i:
                    y1[i] = 0
                    y2[i] = 2
    std = ft.pop_ast()

    assert std.match(ast)
