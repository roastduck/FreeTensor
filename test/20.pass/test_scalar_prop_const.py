import freetensor as ft


def test_basic():
    with ft.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (y1, y2):
        y1[()] = 1
        y2[()] = y1[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (y1, y2):
        y1[()] = 1
        y2[()] = 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_multiple_choices_no_remove():
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.If(x[()] >= 0):
            y1[()] = 1
        with ft.Else():
            y1[()] = 2
        y2[()] = y1[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.If(x[()] >= 0):
            y1[()] = 1
        with ft.Else():
            y1[()] = 2
        y2[()] = y1[()]
    std = ft.pop_ast()

    assert std.match(ast)


def test_multiple_choices_no_remove_2():
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 0
        with ft.For("i", 0, 5) as i:
            y[()] = y[()] + i
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 0
        with ft.For("i", 0, 5) as i:
            y[()] = y[()] + i
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_remove_intermediate_array():
    with ft.VarDef("y", (), "int32", "output", "cpu") as y:
        with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
            t[()] = 2
            y[()] = t[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_non_const_no_remove():
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
            t[()] = x[()] + 1
            y1[()] = t[()] * 2
            y2[()] = t[()] * 3
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
            t[()] = x[()] + 1
            y1[()] = t[()] * 2
            y2[()] = t[()] * 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_propagate():
    with ft.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu"),
                    ("y3", (), "int32", "output", "cpu")]) as (y1, y2, y3):
        y1[()] = 1
        y2[()] = y1[()]
        y3[()] = y2[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu"),
                    ("y3", (), "int32", "output", "cpu")]) as (y1, y2, y3):
        y1[()] = 1
        y2[()] = 1
        y3[()] = 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_propagate_through_expressions():
    with ft.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu"),
                    ("y3", (), "int32", "output", "cpu")]) as (y1, y2, y3):
        y1[()] = 1
        y2[()] = y1[()]
        y3[()] = y2[()] + y2[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu"),
                    ("y3", (), "int32", "output", "cpu")]) as (y1, y2, y3):
        y1[()] = 1
        y2[()] = 1
        y3[()] = 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_prop_iter():
    with ft.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[()] = i
            y2[i] = y1[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.If(i == 3):
                y1[()] = 3
            y2[i] = i
    std = ft.pop_ast()

    assert std.match(ast)


def test_prop_iter_in_expr():
    with ft.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = i + 1
            y2[i] = y1[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.If(i == 3):
                y1[i] = 4
            y2[i] = i + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_prop_iter_in_expr_2():
    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = i + 1
            y2[i] = y1[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = i + 1
            y2[i] = i + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_loop_local_basic():
    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = 1
            y2[i] = y1[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = 1
            y2[i] = 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_loop_local_multiple_choices_no_remove():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.If(x[i] >= 0):
                y1[i] = 1
            with ft.Else():
                y1[i] = 2
            y2[i] = y1[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.If(x[i] >= 0):
                y1[i] = 1
            with ft.Else():
                y1[i] = 2
            y2[i] = y1[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_loop_local_multiple_choices_no_remove_2():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = 0
            with ft.For("j", 0, 5) as j:
                y[i] = y[i] + j
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = 0
            with ft.For("j", 0, 5) as j:
                y[i] = y[i] + j
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_loop_local_prop_iter():
    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = i
            y2[i] = y1[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = i
            y2[i] = i
    std = ft.pop_ast()

    assert std.match(ast)


def test_reduction():
    # This is a case that cannot be handled by pass/remove_writes,
    # so it must be done in prop const

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = 1
            with ft.If(i > 1):
                # This condition stops pass/remove_writes from removing the
                # assignment above
                y[i] += 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['tensor_prop_const'])

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = 1
            with ft.If(i > 1):
                y[i] = 2
    std = ft.pop_ast()

    assert std.match(ast)
