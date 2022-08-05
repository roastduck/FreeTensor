import freetensor as ft


def test_basic():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("b", (4,), "int32", "cache", "cpu") as b:
            b[2] = x[2]
            y1[2] = b[2] + 1
            y2[2] = b[2] + 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("b", (1,), "int32", "cache", "cpu") as b:
            b[0] = x[2]
            y1[2] = b[0] + 1
            y2[2] = b[0] + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_iter():
    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("b", (4,), "int32", "cache", "cpu") as b:
                b[i] = x[i] + x[i + 1]
                y1[i] = b[i] * i
                y2[i] = b[i] + i
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("b", (1,), "int32", "cache", "cpu") as b:
                b[0] = x[i] + x[i + 1]
                y1[i] = b[0] * i
                y2[i] = b[0] + i
    std = ft.pop_ast()

    assert std.match(ast)


def test_multiple_bounds():
    with ft.VarDef([("x", (10, 5), "int32", "input", "cpu"),
                    ("y1", (10, 5), "int32", "output", "cpu"),
                    ("y2", (10, 5), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 10) as i:
            with ft.VarDef("b", (10000,), "int32", "cache", "cpu") as b:
                with ft.For("j", 0, 5) as j:
                    with ft.If(j < i):
                        b[j] = x[i, j] * 2
                with ft.For("j", 0, 5) as j:
                    y1[i, j] = x[i, j] * 2
                    y2[i, j] = x[i, j] * 3
                    with ft.If(j < i):
                        y1[i, j] += b[j]
                        y2[i, j] += b[j]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['make_heap_alloc'])

    with ft.VarDef([("x", (10, 5), "int32", "input", "cpu"),
                    ("y1", (10, 5), "int32", "output", "cpu"),
                    ("y2", (10, 5), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 10) as i:
            with ft.VarDef("b", (ft.min(i - 1, 4) + 1,), "int32", "cache",
                           "cpu") as b:
                with ft.For("j", 0, ft.any()) as j:
                    b[j] = x[i, j] * 2
                with ft.For("j", 0, 5) as j:
                    y1[i, j] = x[i, j] * 2
                    y2[i, j] = x[i, j] * 3
                    with ft.If(j < i):
                        y1[i, j] += b[j]
                        y2[i, j] += b[j]
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_bound_only_on_reads():
    with ft.VarDef([("x", (8,), "int32", "input", "cpu"),
                    ("y1", (8,), "int32", "output", "cpu"),
                    ("y2", (8,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("b", (8,), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 8) as i:
                b[i] = x[i] * 2
            with ft.For("i", 0, 4) as i:
                y1[i] = b[i] + 1
                y2[i] = b[i] + 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (8,), "int32", "input", "cpu"),
                    ("y1", (8,), "int32", "output", "cpu"),
                    ("y2", (8,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("b", (4,), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 4) as i:
                b[i] = x[i] * 2
            with ft.For("i", 0, 4) as i:
                y1[i] = b[i] + 1
                y2[i] = b[i] + 2
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_read_bound_with_offset():
    with ft.VarDef([("x", (8,), "int32", "input", "cpu"),
                    ("y", (6,), "int32", "output", "cpu"),
                    ("z", (8,), "int32", "cache", "cpu")]) as (x, y, z):
        with ft.For("i", 0, 8) as i:
            z[i] = x[i] * 2
        with ft.For("i", 0, 6) as i:
            y[i] = z[i + 1]

    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, skip_passes=['prop_one_time_use'])
    print(ast)

    with ft.VarDef([("x", (8,), "int32", "input", "cpu"),
                    ("y", (6,), "int32", "output", "cpu"),
                    ("z", (6,), "int32", "cache", "cpu")]) as (x, y, z):
        with ft.For("i", 1, 7) as i:
            z[i - 1] = x[i] * 2
        with ft.For("i", 0, 6) as i:
            y[i] = z[i]

    assert ft.pop_ast().match(ast)


# FIXME: Fix this test
#def test_const_in_branch_1():
#    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
#                    ("y1", (4,), "int32", "output", "cpu"),
#                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
#        with ft.For("i", 0, 4) as i:
#            with ft.VarDef("b", (4,), "int32", "cache", "cpu") as b:
#                with ft.If(i == 2):
#                    b[2] = x[2]
#                with ft.Else():
#                    b[i] = x[i] + x[i + 1]
#                y1[i] = b[i] * i
#                y2[i] = b[i] + i
#    ast = ft.pop_ast()
#    print(ast)
#    ast = ft.lower(ast)
#    print(ast)
#
#    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
#                    ("y1", (4,), "int32", "output", "cpu"),
#                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
#        with ft.For("i", 0, 4) as i:
#            with ft.VarDef("b", (1,), "int32", "cache", "cpu") as b:
#                with ft.If(i == 2):
#                    b[0] = x[2]
#                with ft.Else():
#                    b[0] = x[i] + x[i + 1]
#                y1[i] = b[0] * i
#                y2[i] = b[0] + i
#    std = ft.pop_ast()
#
#    assert std.match(ast)

# FIXME: Fix this test
#def test_const_in_branch_2():
#    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
#                    ("y1", (4,), "int32", "output", "cpu"),
#                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
#        with ft.For("i", 0, 4) as i:
#            with ft.VarDef("b", (4,), "int32", "cache", "cpu") as b:
#                with ft.If(i < 3):
#                    b[i] = x[i] + x[i + 1]
#                with ft.Else():
#                    b[3] = x[3]
#                y1[i] = b[i] * i
#                y2[i] = b[i] + i
#    ast = ft.pop_ast()
#    print(ast)
#    ast = ft.lower(ast)
#    print(ast)
#
#    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
#                    ("y1", (4,), "int32", "output", "cpu"),
#                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
#        with ft.For("i", 0, 4) as i:
#            with ft.VarDef("b", (1,), "int32", "cache", "cpu") as b:
#                with ft.If(i < 3):
#                    b[0] = x[i] + x[i + 1]
#                with ft.Else():
#                    b[0] = x[3]
#                y1[i] = b[0] * i
#                y2[i] = b[0] + i
#    std = ft.pop_ast()
#
#    assert std.match(ast)
