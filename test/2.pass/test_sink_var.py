import ir


def test_sink_stmt_seq_back():
    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
            with ir.For("i", 0, 4) as i:
                b[0] = x[i] + x[i + 1]
                y1[i] = b[0] * i
                y2[i] = b[0] + i
            y1[0] = 0
            y2[0] = 0
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
            with ir.For("i", 0, 4) as i:
                b[0] = x[i] + x[i + 1]
                y1[i] = b[0] * i
                y2[i] = b[0] + i
        y1[0] = 0
        y2[0] = 0
    std = ir.pop_ast()

    assert std.match(ast)


def test_sink_stmt_seq_front():
    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
            y1[0] = 0
            y2[0] = 0
            with ir.For("i", 1, 4) as i:
                b[0] = x[i] + x[i + 1]
                y1[i] = b[0] * i
                y2[i] = b[0] + i
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        y1[0] = 0
        y2[0] = 0
        with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
            with ir.For("i", 1, 4) as i:
                b[0] = x[i] + x[i + 1]
                y1[i] = b[0] * i
                y2[i] = b[0] + i
    std = ir.pop_ast()

    assert std.match(ast)


def test_sink_for_no_deps():
    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.VarDef("b", (4,), "int32", "cache", "cpu") as b:
            with ir.For("i", 0, 4) as i:
                b[i] = x[i] + x[i + 1]
                y1[i] = b[i] * i
                y2[i] = b[i] + i
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.For("i", 0, 4) as i:
            # Also shrinked
            with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
                b[0] = x[i] + x[i + 1]
                y1[i] = b[0] * i
                y2[i] = b[0] + i
    std = ir.pop_ast()

    assert std.match(ast)


def test_sink_for_invariant():
    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.VarDef("b", (), "int32", "cache", "cpu") as b:
            with ir.For("i", 0, 4) as i:
                b[()] = x[0] + x[1]
                y1[i] = b[()] * i
                y2[i] = b[()] + i
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("b", (), "int32", "cache", "cpu") as b:
                b[()] = x[0] + x[1]
                y1[i] = b[()] * i
                y2[i] = b[()] + i
    std = ir.pop_ast()

    assert std.match(ast)


def test_cross_other_vardef():
    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.VarDef("b", (4,), "int32", "cache", "cpu") as b:
            with ir.VarDef("t1", (4,), "int32", "cache", "cpu") as t1:
                with ir.VarDef("t2", (4,), "int32", "cache", "cpu") as t2:
                    with ir.For("i", 0, 4) as i:
                        b[i] = x[i] + x[i + 1]
                        t1[i] = b[i] * i
                        t2[i] = b[i] + i
                    with ir.For("i", 0, 4) as i:
                        y1[i] = t1[i] + t2[i]
                        y2[i] = t1[i] * t2[i]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.VarDef("t1", (4,), "int32", "cache", "cpu") as t1:
            with ir.VarDef("t2", (4,), "int32", "cache", "cpu") as t2:
                with ir.For("i", 0, 4) as i:
                    # Also shrinked
                    with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
                        b[0] = x[i] + x[i + 1]
                        t1[i] = b[0] * i
                        t2[i] = b[0] + i
                with ir.For("i", 0, 4) as i:
                    y1[i] = t1[i] + t2[i]
                    y2[i] = t1[i] * t2[i]
    std = ir.pop_ast()

    assert std.match(ast)
