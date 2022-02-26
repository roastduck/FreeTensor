import ir


def test_basic():
    with ir.VarDef("x", (), "int32", "output", "cpu") as x:
        x[()] = 1
    with ir.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.hoist_var_over_stmt_seq(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "output", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        x[()] = 1
        y[()] = 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_rename():
    with ir.VarDef("x", (), "int32", "output", "cpu") as x:
        x[()] = 1
    with ir.VarDef("x", (), "int32", "output", "cpu") as x:
        x[()] = 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.hoist_var_over_stmt_seq(ast)
    print(ast)

    with ir.VarDef([("x.0", (), "int32", "output", "cpu"),
                    ("x.1", (), "int32", "output", "cpu")]) as (x0, x1):
        x0[()] = 1
        x1[()] = 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_rename_nested_1():
    with ir.VarDef("y", (), "int32", "output", "cpu") as y:
        with ir.VarDef("x", (), "int32", "output", "cpu") as x:
            x[()] = 1
        with ir.VarDef("x", (), "int32", "output", "cpu") as x:
            x[()] = 2
        y[()] = 0
    with ir.VarDef("x", (), "int32", "output", "cpu") as x:
        x[()] = 3
    ast = ir.pop_ast()
    print(ast)
    ast = ir.hoist_var_over_stmt_seq(ast)
    print(ast)

    with ir.VarDef([("y", (), "int32", "output", "cpu"),
                    ("x", (), "int32", "output", "cpu"),
                    ("x.0", (), "int32", "output", "cpu"),
                    ("x.1", (), "int32", "output", "cpu")]) as (y, x, x0, x1):
        x0[()] = 1
        x1[()] = 2
        y[()] = 0
        x[()] = 3
    std = ir.pop_ast()

    assert std.match(ast)


def test_rename_nested_2():
    with ir.VarDef("x", (), "int32", "output", "cpu") as x:
        x[()] = 1
    with ir.VarDef("y", (), "int32", "output", "cpu") as y:
        with ir.VarDef("x", (), "int32", "output", "cpu") as x:
            x[()] = 2
        y[()] = 0
    ast = ir.pop_ast()
    print(ast)
    ast = ir.hoist_var_over_stmt_seq(ast)
    print(ast)

    with ir.VarDef([("x.0", (), "int32", "output", "cpu"),
                    ("y", (), "int32", "output", "cpu"),
                    ("x.1", (), "int32", "output", "cpu")]) as (x0, y, x1):
        x0[()] = 1
        x1[()] = 2
        y[()] = 0
    std = ir.pop_ast()

    assert std.match(ast)
