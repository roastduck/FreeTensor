import ir


def test_basic():
    with ir.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ir.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
        y[()] = x[()] + 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_chained():
    with ir.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ir.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
            with ir.VarDef("b", (), "int32", "cache", "cpu") as b:
                b[()] = a[()] + 1
                with ir.VarDef("c", (), "int32", "cache", "cpu") as c:
                    c[()] = a[()] + b[()]
        y[()] = x[()] + 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_self_assign():
    with ir.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ir.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = 0
            with ir.For("i", 0, 10) as i:
                a[()] = a[()] * x[()] + 1
        y[()] = x[()] + 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "inout", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] + 1
    std = ir.pop_ast()

    assert std.match(ast)
