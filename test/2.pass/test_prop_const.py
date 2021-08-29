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
