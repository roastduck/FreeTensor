import ir


def test_basic():
    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            y1[i] = 1
            y2[i] = y1[i]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            y1[i] = 1
            y2[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_multiple_choices_no_remove():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.If(x[i] >= 0):
                y1[i] = 1
            with ir.Else():
                y1[i] = 2
            y2[i] = y1[i]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.If(x[i] >= 0):
                y1[i] = 1
            with ir.Else():
                y1[i] = 2
            y2[i] = y1[i]
    std = ir.pop_ast()

    assert std.match(ast)


def test_multiple_choices_no_remove_2():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = 0
            with ir.For("j", 0, 5) as j:
                y[i] = y[i] + j
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = 0
            with ir.For("j", 0, 5) as j:
                y[i] = y[i] + j
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_propagate():
    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu"),
                    ("y3", (4,), "int32", "output", "cpu")]) as (y1, y2, y3):
        with ir.For("i", 0, 4) as i:
            y1[i] = 1
        with ir.For("i", 0, 4) as i:
            y2[i] = y1[i]
        with ir.For("i", 0, 4) as i:
            y3[i] = y2[i]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu"),
                    ("y3", (4,), "int32", "output", "cpu")]) as (y1, y2, y3):
        with ir.For("i", 0, 4) as i:
            y1[i] = 1
        with ir.For("i", 0, 4) as i:
            y2[i] = 1
        with ir.For("i", 0, 4) as i:
            y3[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_propagate_through_expressions():
    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu"),
                    ("y3", (4,), "int32", "output", "cpu")]) as (y1, y2, y3):
        with ir.For("i", 0, 4) as i:
            y1[i] = 1
        with ir.For("i", 0, 4) as i:
            y2[i] = y1[i]
        with ir.For("i", 0, 4) as i:
            y3[i] = y2[i] + y2[i]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu"),
                    ("y3", (4,), "int32", "output", "cpu")]) as (y1, y2, y3):
        with ir.For("i", 0, 4) as i:
            y1[i] = 1
        with ir.For("i", 0, 4) as i:
            y2[i] = 1
        with ir.For("i", 0, 4) as i:
            y3[i] = 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_prop_iter():
    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            y1[i] = i
            y2[i] = y1[i]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            y1[i] = i
            y2[i] = i
    std = ir.pop_ast()

    assert std.match(ast)


def test_prop_iter_different_scope_no_prop():
    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            y1[i] = i
        with ir.For("i", 2, 4) as i:
            y2[i] = y1[i]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            y1[i] = i
        with ir.For("i", 2, 4) as i:
            y2[i] = y1[i]
    std = ir.pop_ast()

    assert std.match(ast)


def test_prop_iter_different_instance_no_prop():
    with ir.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.If(i % 2 == 0):
                y1[()] = i
            y2[i] = y1[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.If(i % 2 == 0):
                y1[()] = i
            y2[i] = y1[()]
    std = ir.use_builtin_div(ir.pop_ast())

    assert std.match(ast)


def test_prop_iter_in_expr():
    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            y1[i] = i + 1
            y2[i] = y1[i]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            y1[i] = i + 1
            y2[i] = i + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_reduction():
    # This is a case that cannot be handled by pass/remove_writes,
    # so it must be done in pass/tensor_prop_const

    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = 1
            with ir.If(i > 1):
                # This condition stops pass/remove_writes from removing the
                # assignment above
                y[i] += 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = 1
            with ir.If(i > 1):
                y[i] = 2
    std = ir.pop_ast()

    assert std.match(ast)
