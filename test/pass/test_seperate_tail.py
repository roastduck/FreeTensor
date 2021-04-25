import ir

def test_basic():
    with ir.VarDef([
            ("y1", (4,), "int32", "output", "cpu"),
            ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.If(i < 2):
                y1[i] = 0
            with ir.Else():
                y1[i] = 1
            with ir.If(i < 2):
                y2[i] = 2
            with ir.Else():
                y2[i] = 3
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("y1", (4,), "int32", "output", "cpu"),
            ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 2) as i:
            y1[i] = 0
            y2[i] = 2
        with ir.For("i", 2, 4) as i:
            y1[i] = 1
            y2[i] = 3
    std = ir.pop_ast()

    assert std.match(ast)

def test_multiple_cond():
    with ir.VarDef([
            ("y1", (5,), "int32", "output", "cpu"),
            ("y2", (5,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 5) as i:
            with ir.If(i < 2):
                y1[i] = 0
            with ir.Else():
                y1[i] = 1
            with ir.If(i < 3):
                y2[i] = 2
            with ir.Else():
                y2[i] = 3
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("y1", (5,), "int32", "output", "cpu"),
            ("y2", (5,), "int32", "output", "cpu")]) as (y1, y2):
        with ir.For("i", 0, 2) as i:
            y1[i] = 0
            y2[i] = 2
        y1[2] = 1
        y2[2] = 2
        with ir.For("i", 3, 5) as i:
            y1[i] = 1
            y2[i] = 3
    std = ir.pop_ast()

    assert std.match(ast)

def test_eq():
    with ir.VarDef("y", (5,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 5) as i:
            with ir.If(i == 2):
                y[i] = 0
            with ir.Else():
                y[i] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (5,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 2) as i:
            y[i] = 1
        y[2] = 0
        with ir.For("i", 3, 5) as i:
            y[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)

def test_tiled():
    with ir.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 3) as i:
            with ir.For("j", 0, 4) as j:
                with ir.If(4 * i + j < 10):
                    y[4 * i + j] = 4 * i + j
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 2) as i:
            with ir.For("j", 0, 4) as j:
                y[4 * i + j] = 4 * i + j
        with ir.For("j", 0, 2) as j:
            y[8 + j] = 8 + j
    std = ir.pop_ast()

    assert std.match(ast)

def test_dynamic_tiled():
    with ir.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ir.Assert(n[()] > 0):
            with ir.VarDef("y", (n[()],), "int32", "output", "cpu") as y:
                with ir.For("i", 0, ir.ceil_div(n[()], 4)) as i:
                    with ir.For("j", 0, 4) as j:
                        with ir.If(4 * i + j < n[()]):
                            y[4 * i + j] = 4 * i + j
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("n", (), "int32", "input", "byvalue") as n:
        with ir.Assert(n[()] > 0):
            with ir.VarDef("y", (n[()],), "int32", "output", "cpu") as y:
                with ir.For("i", 0, ir.any()) as i:
                    with ir.For("j", 0, 4) as j:
                        y[4 * i + j] = 4 * i + j
                with ir.For("j", 0, ir.any()) as j:
                    y[ir.any() + j] = ir.any() + j
    std = ir.pop_ast()

    assert std.match(ast)

def test_1d_stencil():
    with ir.VarDef([
            ("x", (10,), "int32", "input", "cpu"),
            ("y", (10,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 10) as i:
            y[i] = x[i]
            with ir.If(i - 1 >= 0):
                y[i] = y[i] + x[i - 1]
            with ir.If(i + 1 < 10):
                y[i] = y[i] + x[i + 1]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (10,), "int32", "input", "cpu"),
            ("y", (10,), "int32", "output", "cpu")]) as (x, y):
        y[0] = x[0] + x[1]
        with ir.For("i", 1, 9) as i:
            y[i] = x[i + -1] + (x[i] + x[i + 1])
        y[9] = x[9] + x[8]
    std = ir.pop_ast()

    assert std.match(ast)

