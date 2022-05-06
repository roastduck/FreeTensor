import freetensor as ft


def test_basic():
    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = 1
            y2[i] = y1[i]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = 1
            y2[i] = 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_multiple_choices_no_remove():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.If(x[i] >= 0):
                y1[i] = 1
            with ft.Else():
                y1[i] = 2
            y2[i] = y1[i]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

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


def test_multiple_choices_no_remove_2():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = 0
            with ft.For("j", 0, 5) as j:
                y[i] = y[i] + j
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = 0
            with ft.For("j", 0, 5) as j:
                y[i] = y[i] + j
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_propagate():
    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu"),
                    ("y3", (4,), "int32", "output", "cpu")]) as (y1, y2, y3):
        with ft.For("i", 0, 4) as i:
            y1[i] = 1
        with ft.For("i", 0, 4) as i:
            y2[i] = y1[i]
        with ft.For("i", 0, 4) as i:
            y3[i] = y2[i]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu"),
                    ("y3", (4,), "int32", "output", "cpu")]) as (y1, y2, y3):
        with ft.For("i", 0, 4) as i:
            y1[i] = 1
        with ft.For("i", 0, 4) as i:
            y2[i] = 1
        with ft.For("i", 0, 4) as i:
            y3[i] = 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_propagate_through_expressions():
    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu"),
                    ("y3", (4,), "int32", "output", "cpu")]) as (y1, y2, y3):
        with ft.For("i", 0, 4) as i:
            y1[i] = 1
        with ft.For("i", 0, 4) as i:
            y2[i] = y1[i]
        with ft.For("i", 0, 4) as i:
            y3[i] = y2[i] + y2[i]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu"),
                    ("y3", (4,), "int32", "output", "cpu")]) as (y1, y2, y3):
        with ft.For("i", 0, 4) as i:
            y1[i] = 1
        with ft.For("i", 0, 4) as i:
            y2[i] = 1
        with ft.For("i", 0, 4) as i:
            y3[i] = 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_prop_iter():
    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = i
            y2[i] = y1[i]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = i
            y2[i] = i
    std = ft.pop_ast()

    assert std.match(ast)


def test_prop_iter_different_scope():
    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = i
        with ft.For("i", 2, 4) as i:
            y2[i] = y1[i]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = i
        with ft.For("i", 2, 4) as i:
            y2[i] = i
    std = ft.pop_ast()

    assert std.match(ast)


def test_prop_iter_different_iter():
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = i * 2
            with ft.For("i", 4, 8) as i:
                y[i + -4] = t[i + -4] + 1
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 4, 8) as i:
            y[i + -4] = i * 2 + -7
    std = ft.pop_ast()

    assert std.match(ast)


def test_prop_iter_different_iter_non_linear():
    with ft.VarDef("y", (16,), "int32", "output", "cpu") as y:
        ft.MarkNid("T")
        with ft.VarDef("t", (16,), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                with ft.For("j", 0, 4) as j:
                    t[i * 4 + j] = i
            with ft.For("k", 0, 16) as k:
                y[k] = t[k]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=["use_builtin_div"])
    print(ast)

    with ft.VarDef("y", (16,), "int32", "output", "cpu") as y:
        with ft.For("k", 0, 16) as k:
            y[k] = k // 4
    std = ft.pop_ast()

    assert std.match(ast)


def test_prop_iter_different_instance_no_prop():
    with ft.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.If(i % 2 == 0):
                y1[()] = i
            y2[i] = y1[()]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=["use_builtin_div"])
    print(ast)

    with ft.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.If(i % 2 == 0):
                y1[()] = i
            y2[i] = y1[()]
    std = ft.pop_ast()

    assert std.match(ast)


def test_prop_iter_in_expr():
    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = i + 1
            y2[i] = y1[i]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            y1[i] = i + 1
            y2[i] = i + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_reduction():
    # This is a case that cannot be handled by pass/remove_writes,
    # so it must be done in pass/tensor_prop_const

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = 1
            with ft.If(i > 1):
                # This condition stops pass/remove_writes from removing the
                # assignment above
                y[i] += 1
    ast = ft.pop_ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = 1
            with ft.If(i > 1):
                y[i] = 2
    std = ft.pop_ast()

    assert std.match(ast)
