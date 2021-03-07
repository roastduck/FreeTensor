import ir

def test_basic():
    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.VarDef("b", (4,), "int32", "cache", "cpu") as b:
            b[2] = x[2]
            y[2] = b[2]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
            b[0] = x[2]
            y[2] = b[0]
    std = ir.pop_ast()

    assert std.match(ast)

