import ir


def test_basic():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.VarDef("b", (4,), "int32", "cache", "cpu") as b:
            b[2] = x[2]
            y1[2] = b[2] + 1
            y2[2] = b[2] + 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
            b[0] = x[2]
            y1[2] = b[0] + 1
            y2[2] = b[0] + 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_iter():
    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y1", (4,), "int32", "output", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("b", (4,), "int32", "cache", "cpu") as b:
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
            with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
                b[0] = x[i] + x[i + 1]
                y1[i] = b[0] * i
                y2[i] = b[0] + i
    std = ir.pop_ast()

    assert std.match(ast)


def test_multiple_bounds():
    with ir.VarDef([("x", (10, 5), "int32", "input", "cpu"),
                    ("y1", (10, 5), "int32", "output", "cpu"),
                    ("y2", (10, 5), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.For("i", 0, 10) as i:
            with ir.VarDef("b", (10000,), "int32", "cache", "cpu") as b:
                with ir.For("j", 0, 5) as j:
                    with ir.If(j < i):
                        b[j] = x[i, j] * 2
                with ir.For("j", 0, 5) as j:
                    y1[i, j] = x[i, j] * 2
                    y2[i, j] = x[i, j] * 3
                    with ir.If(j < i):
                        y1[i, j] += b[j]
                        y2[i, j] += b[j]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (10, 5), "int32", "input", "cpu"),
                    ("y1", (10, 5), "int32", "output", "cpu"),
                    ("y2", (10, 5), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.For("i", 0, 10) as i:
            with ir.VarDef("b", (ir.min(i - 1, 4) + 1,), "int32", "cache",
                           "cpu") as b:
                with ir.For("j", 0, ir.any()) as j:
                    b[j] = x[i, j] * 2
                with ir.For("j", 0, 5) as j:
                    y1[i, j] = x[i, j] * 2
                    y2[i, j] = x[i, j] * 3
                    with ir.If(j < i):
                        y1[i, j] += b[j]
                        y2[i, j] += b[j]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


# FIXME: Fix this test
#def test_const_in_branch_1():
#    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
#                    ("y1", (4,), "int32", "output", "cpu"),
#                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
#        with ir.For("i", 0, 4) as i:
#            with ir.VarDef("b", (4,), "int32", "cache", "cpu") as b:
#                with ir.If(i == 2):
#                    b[2] = x[2]
#                with ir.Else():
#                    b[i] = x[i] + x[i + 1]
#                y1[i] = b[i] * i
#                y2[i] = b[i] + i
#    ast = ir.pop_ast()
#    print(ast)
#    ast = ir.lower(ast)
#    print(ast)
#
#    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
#                    ("y1", (4,), "int32", "output", "cpu"),
#                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
#        with ir.For("i", 0, 4) as i:
#            with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
#                with ir.If(i == 2):
#                    b[0] = x[2]
#                with ir.Else():
#                    b[0] = x[i] + x[i + 1]
#                y1[i] = b[0] * i
#                y2[i] = b[0] + i
#    std = ir.pop_ast()
#
#    assert std.match(ast)

# FIXME: Fix this test
#def test_const_in_branch_2():
#    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
#                    ("y1", (4,), "int32", "output", "cpu"),
#                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
#        with ir.For("i", 0, 4) as i:
#            with ir.VarDef("b", (4,), "int32", "cache", "cpu") as b:
#                with ir.If(i < 3):
#                    b[i] = x[i] + x[i + 1]
#                with ir.Else():
#                    b[3] = x[3]
#                y1[i] = b[i] * i
#                y2[i] = b[i] + i
#    ast = ir.pop_ast()
#    print(ast)
#    ast = ir.lower(ast)
#    print(ast)
#
#    with ir.VarDef([("x", (5,), "int32", "input", "cpu"),
#                    ("y1", (4,), "int32", "output", "cpu"),
#                    ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
#        with ir.For("i", 0, 4) as i:
#            with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
#                with ir.If(i < 3):
#                    b[0] = x[i] + x[i + 1]
#                with ir.Else():
#                    b[0] = x[3]
#                y1[i] = b[0] * i
#                y2[i] = b[0] + i
#    std = ir.pop_ast()
#
#    assert std.match(ast)
