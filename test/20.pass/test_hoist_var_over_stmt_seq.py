import freetensor as ft


def test_basic():
    with ft.VarDef("x", (), "int32", "cache", "cpu") as x:
        x[()] = 1
    with ft.VarDef("y", (), "int32", "cache", "cpu") as y:
        y[()] = 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.hoist_var_over_stmt_seq(ast)
    print(ast)

    with ft.VarDef([("x", (), "int32", "cache", "cpu"),
                    ("y", (), "int32", "cache", "cpu")]) as (x, y):
        x[()] = 1
        y[()] = 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_rename():
    with ft.VarDef("x", (), "int32", "cache", "cpu") as x:
        x[()] = 1
    with ft.VarDef("x", (), "int32", "cache", "cpu") as x:
        x[()] = 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.hoist_var_over_stmt_seq(ast)
    print(ast)

    with ft.VarDef([("x.0", (), "int32", "cache", "cpu"),
                    ("x.1", (), "int32", "cache", "cpu")]) as (x0, x1):
        x0[()] = 1
        x1[()] = 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_rename_nested_1():
    with ft.VarDef("y", (), "int32", "cache", "cpu") as y:
        with ft.VarDef("x", (), "int32", "cache", "cpu") as x:
            x[()] = 1
        with ft.VarDef("x", (), "int32", "cache", "cpu") as x:
            x[()] = 2
        y[()] = 0
    with ft.VarDef("x", (), "int32", "cache", "cpu") as x:
        x[()] = 3
    ast = ft.pop_ast(verbose=True)
    ast = ft.hoist_var_over_stmt_seq(ast)
    print(ast)

    with ft.VarDef([("y", (), "int32", "cache", "cpu"),
                    ("x", (), "int32", "cache", "cpu"),
                    ("x.0", (), "int32", "cache", "cpu"),
                    ("x.1", (), "int32", "cache", "cpu")]) as (y, x, x0, x1):
        x0[()] = 1
        x1[()] = 2
        y[()] = 0
        x[()] = 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_rename_nested_2():
    with ft.VarDef("x", (), "int32", "cache", "cpu") as x:
        x[()] = 1
    with ft.VarDef("y", (), "int32", "cache", "cpu") as y:
        with ft.VarDef("x", (), "int32", "cache", "cpu") as x:
            x[()] = 2
        y[()] = 0
    ast = ft.pop_ast(verbose=True)
    ast = ft.hoist_var_over_stmt_seq(ast)
    print(ast)

    with ft.VarDef([("x.0", (), "int32", "cache", "cpu"),
                    ("y", (), "int32", "cache", "cpu"),
                    ("x.1", (), "int32", "cache", "cpu")]) as (x0, y, x1):
        x0[()] = 1
        x1[()] = 2
        y[()] = 0
    std = ft.pop_ast()

    assert std.match(ast)


def test_no_hoisting_affected_shape_front():
    with ft.VarDef("n", (), "int32", "input-mutable", "cpu") as n:
        n[...] += 1
        with ft.VarDef([("x", (n[...],), "int32", "cache", "cpu"),
                        ("y", (n[...],), "int32", "cache", "cpu")]) as (x, y):
            with ft.For("i", 0, n[...]) as i:
                y[i] = x[i] + 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.hoist_var_over_stmt_seq(ast)
    print(ast)

    with ft.VarDef("n", (), "int32", "input-mutable", "cpu") as n:
        n[...] += 1
        with ft.VarDef([("x", (n[...],), "int32", "cache", "cpu"),
                        ("y", (n[...],), "int32", "cache", "cpu")]) as (x, y):
            with ft.For("i", 0, n[...]) as i:
                y[i] = x[i] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_no_hoisting_affected_shape_back():
    with ft.VarDef("n", (), "int32", "input-mutable", "cpu") as n:
        with ft.VarDef([("x", (n[...],), "int32", "cache", "cpu"),
                        ("y", (n[...],), "int32", "cache", "cpu")]) as (x, y):
            with ft.For("i", 0, n[...]) as i:
                y[i] = x[i] + 1
        n[...] += 1
        # We can't hoist `x` or `y` even `n += 1` is at the back, beacuse we
        # forbit modifying the shape inside its VarDef
    ast = ft.pop_ast(verbose=True)
    ast = ft.hoist_var_over_stmt_seq(ast)
    print(ast)

    with ft.VarDef("n", (), "int32", "input-mutable", "cpu") as n:
        with ft.VarDef([("x", (n[...],), "int32", "cache", "cpu"),
                        ("y", (n[...],), "int32", "cache", "cpu")]) as (x, y):
            with ft.For("i", 0, n[...]) as i:
                y[i] = x[i] + 1
        n[...] += 1
    std = ft.pop_ast()

    assert std.match(ast)
