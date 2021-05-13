import ir

def test_merge():
    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y1", (4,), "int32", "output", "cpu"),
            ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.If(x[i] < 2):
                y1[i] = 0
            with ir.Else():
                y1[i] = 1
            with ir.If(x[i] < 2):
                y2[i] = 2
            with ir.Else():
                y2[i] = 3
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y1", (4,), "int32", "output", "cpu"),
            ("y2", (4,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.If(x[i] < 2):
                y1[i] = 0
                y2[i] = 2
            with ir.Else():
                y1[i] = 1
                y2[i] = 3
    std = ir.pop_ast()

    assert std.match(ast)

def test_no_merge_different_cond():
    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y1", (5,), "int32", "output", "cpu"),
            ("y2", (5,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.For("i", 0, 5) as i:
            with ir.If(x[i] < 2):
                y1[i] = 0
            with ir.Else():
                y1[i] = 1
            with ir.If(x[i] < 3):
                y2[i] = 2
            with ir.Else():
                y2[i] = 3
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y1", (5,), "int32", "output", "cpu"),
            ("y2", (5,), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.For("i", 0, 5) as i:
            with ir.If(x[i] < 2):
                y1[i] = 0
            with ir.Else():
                y1[i] = 1
            with ir.If(x[i] < 3):
                y2[i] = 2
            with ir.Else():
                y2[i] = 3
    std = ir.pop_ast()

    assert std.match(ast)

def test_no_merge_may_update():
    with ir.VarDef("a", (4,), "int32", "inout", "cpu") as a:
        with ir.For("i", 0, 4) as i:
            with ir.If(a[i] > 10):
                a[i] = a[i] / 2
            with ir.If(a[i] > 10):
                a[i] = a[i] / 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("a", (4,), "int32", "inout", "cpu") as a:
        with ir.For("i", 0, 4) as i:
            with ir.If(a[i] > 10):
                a[i] = a[i] / 2
            with ir.If(a[i] > 10):
                a[i] = a[i] / 2
    std = ir.pop_ast()

    assert std.match(ast)

def test_hoist():
    with ir.VarDef([
            ("x", (4, 4), "int32", "input", "cpu"),
            ("y", (4, 4), "int32", "inout", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                with ir.If(i % 2 == 0):
                    y[i, j] = x[i, j]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (4, 4), "int32", "input", "cpu"),
            ("y", (4, 4), "int32", "inout", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(i % 2 == 0):
                with ir.For("j", 0, 4) as j:
                    y[i, j] = x[i, j]
    std = ir.pop_ast()

    assert std.match(ast)

def test_not_hoisting_not_pure_nested():
    with ir.VarDef([
            ("x", (4, 4), "int32", "input", "cpu"),
            ("y", (4, 4), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                y[i, j] = 0
                with ir.If(i % 2 == 0):
                    y[i, j] = x[i, j]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (4, 4), "int32", "input", "cpu"),
            ("y", (4, 4), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                y[i, j] = 0
                with ir.If(i % 2 == 0):
                    y[i, j] = x[i, j]
    std = ir.pop_ast()

    assert std.match(ast)

def test_not_hoisting_when_being_updated():
    with ir.VarDef([
            ("n", (), "int32", "inout", "cpu"),
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "inout", "cpu")]) as (n, x, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(n[()] < x[i]):
                y[i] = 0
                n[()] = n[()] + x[i]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("n", (), "int32", "inout", "cpu"),
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "inout", "cpu")]) as (n, x, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(n[()] < x[i]):
                y[i] = 0
                n[()] = n[()] + x[i]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)

def test_hoist_then_merge():
    with ir.VarDef([
            ("x", (4, 4), "int32", "input", "cpu"),
            ("y", (4, 4), "int32", "inout", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                with ir.If(i % 2 == 0):
                    y[i, j] = x[i, j]
            with ir.If(i % 2 == 0):
                y[i, 0] = y[i, 0] + 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (4, 4), "int32", "input", "cpu"),
            ("y", (4, 4), "int32", "inout", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(i % 2 == 0):
                with ir.For("j", 0, 4) as j:
                    y[i, j] = x[i, j]
                y[i, 0] = y[i, 0] + 1
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)

def test_merge_then_hoist():
    with ir.VarDef([
            ("x", (), "int32", "input", "cpu"),
            ("y1", (4,), "int32", "inout", "cpu"),
            ("y2", (4,), "int32", "inout", "cpu")]) as (x, y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.If(x[()] < 2):
                y1[i] = 0
            with ir.If(x[()] < 2):
                y2[i] = 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("x", (), "int32", "input", "cpu") as x:
        with ir.If(x[()] < 2):
            with ir.VarDef([
                    ("y1", (4,), "int32", "inout", "cpu"),
                    ("y2", (4,), "int32", "inout", "cpu")]) as (y1, y2):
                with ir.For("i", 0, 4) as i:
                    y1[i] = 0
                    y2[i] = 2
    std = ir.pop_ast()

    assert std.match(ast)

