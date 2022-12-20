import freetensor as ft


def test_sink_stmt_seq_back():
    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("b", (1,), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 4) as i:
                b[0] = x[i] + x[i + 1]
                y1[i] = b[0] * i
                y2[i] = b[0] + i
            y1[0] = 0
            y2[0] = 0
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("b", (1,), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 4) as i:
                b[0] = x[i] + x[i + 1]
                y1[i] = b[0] * i
                y2[i] = b[0] + i
        y1[0] = 0
        y2[0] = 0
    std = ft.pop_ast()

    assert std.match(ast)


def test_sink_stmt_seq_front():
    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("b", (1,), "int32", "cache", "cpu") as b:
            y1[0] = 0
            y2[0] = 0
            with ft.For("i", 1, 4) as i:
                b[0] = x[i] + x[i + 1]
                y1[i] = b[0] * i
                y2[i] = b[0] + i
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        y1[0] = 0
        y2[0] = 0
        with ft.VarDef("b", (1,), "int32", "cache", "cpu") as b:
            with ft.For("i", 1, 4) as i:
                b[0] = x[i] + x[i + 1]
                y1[i] = b[0] * i
                y2[i] = b[0] + i
    std = ft.pop_ast()

    assert std.match(ast)


def test_sink_for_no_deps():
    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("b", (4,), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 4) as i:
                b[i] = x[i] + x[i + 1]
                y1[i] = b[i] * i
                y2[i] = b[i] + i
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            # Also shrinked
            with ft.VarDef("b", (1,), "int32", "cache", "cpu") as b:
                b[0] = x[i] + x[i + 1]
                y1[i] = b[0] * i
                y2[i] = b[0] + i
    std = ft.pop_ast()

    assert std.match(ast)


def test_sink_for_invariant():
    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("b", (), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 4) as i:
                b[()] = x[0] + x[1]
                y1[i] = b[()] * i
                y2[i] = b[()] + i
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("b", (), "int32", "cache", "cpu") as b:
                b[()] = x[0] + x[1]
                y1[i] = b[()] * i
                y2[i] = b[()] + i
    std = ft.pop_ast()

    assert std.match(ast)


def test_no_sink_reduction():
    with ft.VarDef("y", (32,), "int32", "output", "cpu") as y:
        with ft.VarDef("b", (), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 32) as i:
                with ft.If(i % 4 == 0):
                    b[()] = 0
                b[()] += 1
                y[i] = b[()] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['use_builtin_div'])

    with ft.VarDef("y", (32,), "int32", "output", "cpu") as y:
        with ft.VarDef("b", (), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 32) as i:
                with ft.If(i % 4 == 0):
                    b[()] = 0
                b[()] += 1
                y[i] = b[()] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_sink_if():
    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("b", (4,), "int32", "cache", "cpu") as b:
                with ft.If(i == 2):
                    b[i] = x[i] + x[i + 1] + 1
                    y1[i] = b[i] * i
                    y2[i] = b[i] + i
                with ft.Else():
                    b[i] = x[i] + x[i + 1]
                    y1[i] = b[i] * i
                    y2[i] = b[i] + i
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.If(i == 2):
                # Also shrinked
                with ft.VarDef("b", (1,), "int32", "cache", "cpu") as b:
                    b[0] = x[2] + x[3] + 1
                    y1[2] = b[0] * 2
                    y2[2] = b[0] + 2
            with ft.Else():
                # Also shrinked
                with ft.VarDef("b", (1,), "int32", "cache", "cpu") as b:
                    b[0] = x[i] + x[i + 1]
                    y1[i] = b[0] * i
                    y2[i] = b[0] + i
    std = ft.pop_ast()

    assert std.match(ast)


def test_cross_other_vardef():
    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("b", (4,), "int32", "cache", "cpu") as b:
            with ft.VarDef("t1", (4,), "int32", "cache", "cpu") as t1:
                with ft.VarDef("t2", (4,), "int32", "cache", "cpu") as t2:
                    with ft.For("i", 0, 4) as i:
                        b[i] = x[i] + x[i + 1]
                        t1[i] = b[i] * i
                        t2[i] = b[i] + i
                    with ft.For("i", 0, 4) as i:
                        y1[i] = t1[i] + t2[i]
                        y2[i] = t1[i] * t2[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("t1", (4,), "int32", "cache", "cpu") as t1:
            with ft.VarDef("t2", (4,), "int32", "cache", "cpu") as t2:
                with ft.For("i", 0, 4) as i:
                    # Also shrinked
                    with ft.VarDef("b", (1,), "int32", "cache", "cpu") as b:
                        b[0] = x[i] + x[i + 1]
                        t1[i] = b[0] * i
                        t2[i] = b[0] + i
                with ft.For("i", 0, 4) as i:
                    y1[i] = t1[i] + t2[i]
                    y2[i] = t1[i] * t2[i]
    std = ft.pop_ast()

    assert std.match(ast)
