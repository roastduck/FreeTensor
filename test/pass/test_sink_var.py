import ir

def test_sink_stmt_seq():
    with ir.VarDef([
            ("x", (5,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
            with ir.For("i", 0, 4) as i:
                b[0] = x[i] + x[i + 1]
                y[i] = b[0] * i
            y[0] = 0
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (5,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
            with ir.For("i", 0, 4) as i:
                b[0] = x[i] + x[i + 1]
                y[i] = b[0] * i
        y[0] = 0
    std = ir.pop_ast()

    assert std.match(ast)

def test_sink_for():
    with ir.VarDef([
            ("x", (5,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.VarDef("b", (4,), "int32", "cache", "cpu") as b:
            with ir.For("i", 0, 4) as i:
                b[i] = x[i] + x[i + 1]
                y[i] = b[i] * i
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (5,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            # Also shrinked
            with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
                b[0] = x[i] + x[i + 1]
                y[i] = b[0] * i
    std = ir.pop_ast()

    assert std.match(ast)

