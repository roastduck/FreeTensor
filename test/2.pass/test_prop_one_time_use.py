import ir


def test_basic():
    with ir.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("t", (4,), "int32", "cache", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, t, y):
        with ir.For("i", 0, 4) as i:
            t[i] = x[i] + 1
            y[i] = t[i] * 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_used_in_many_stmts_no_prop():
    with ir.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("t", (4,), "int32", "cache", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, t, y1, y2):
        with ir.For("i", 0, 4) as i:
            t[i] = x[i] + 1
            y1[i] = t[i] * 2
            y2[i] = t[i] * 3
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("t", (4,), "int32", "cache", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, t, y1, y2):
        with ir.For("i", 0, 4) as i:
            t[i] = x[i] + 1
            y1[i] = t[i] * 2
            y2[i] = t[i] * 3
    std = ir.pop_ast()

    assert std.match(ast)


def test_used_in_many_reads_no_prop():
    with ir.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("t", (4,), "int32", "cache", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, t, y):
        with ir.For("i", 0, 4) as i:
            t[i] = x[i] + 1
            y[i] = t[i] * t[i] + t[i]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "output", "cpu"),
                    ("t", (4,), "int32", "cache", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, t, y):
        with ir.For("i", 0, 4) as i:
            t[i] = x[i] + 1
            y[i] = t[i] * t[i] + t[i]
    std = ir.pop_ast()

    assert std.match(ast)
