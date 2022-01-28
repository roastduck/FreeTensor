import ir


def test_basic():
    with ir.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (y1, y2):
        y1[()] = 1
        y2[()] = y1[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (y1, y2):
        y1[()] = 1
        y2[()] = 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_multiple_choices_no_remove():
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.If(x[()] >= 0):
            y1[()] = 1
        with ir.Else():
            y1[()] = 2
        y2[()] = y1[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.If(x[()] >= 0):
            y1[()] = 1
        with ir.Else():
            y1[()] = 2
        y2[()] = y1[()]
    std = ir.pop_ast()

    assert std.match(ast)


def test_multiple_choices_no_remove_2():
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 0
        with ir.For("i", 0, 5) as i:
            y[()] = y[()] + i
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 0
        with ir.For("i", 0, 5) as i:
            y[()] = y[()] + i
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_remove_intermediate_array():
    with ir.VarDef("y", (), "int32", "output", "cpu") as y:
        with ir.VarDef("t", (), "int32", "cache", "cpu") as t:
            t[()] = 2
            y[()] = t[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_non_const_no_remove():
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.VarDef("t", (), "int32", "cache", "cpu") as t:
            t[()] = x[()] + 1
            y1[()] = t[()] * 2
            y2[()] = t[()] * 3
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.VarDef("t", (), "int32", "cache", "cpu") as t:
            t[()] = x[()] + 1
            y1[()] = t[()] * 2
            y2[()] = t[()] * 3
    std = ir.pop_ast()

    assert std.match(ast)


def test_propagate():
    with ir.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu"),
                    ("y3", (), "int32", "output", "cpu")]) as (y1, y2, y3):
        y1[()] = 1
        y2[()] = y1[()]
        y3[()] = y2[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu"),
                    ("y3", (), "int32", "output", "cpu")]) as (y1, y2, y3):
        y1[()] = 1
        y2[()] = 1
        y3[()] = 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_propagate_through_expressions():
    with ir.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu"),
                    ("y3", (), "int32", "output", "cpu")]) as (y1, y2, y3):
        y1[()] = 1
        y2[()] = y1[()]
        y3[()] = y2[()] + y2[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu"),
                    ("y3", (), "int32", "output", "cpu")]) as (y1, y2, y3):
        y1[()] = 1
        y2[()] = 1
        y3[()] = 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_prop_iter():
    with ir.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            y1[()] = i
            y2[i] = y1[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.If(i == 3):
                y1[()] = 3
            y2[i] = i
    std = ir.pop_ast()

    assert std.match(ast)


def test_prop_iter_in_expr():
    with ir.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            y1[i] = i + 1
            y2[i] = y1[i]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y1", (), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.If(i == 3):
                y1[i] = 4
            y2[i] = i + 1
    std = ir.pop_ast()

    assert std.match(ast)
