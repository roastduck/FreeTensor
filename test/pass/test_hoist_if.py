import ir

def test_basic():
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

