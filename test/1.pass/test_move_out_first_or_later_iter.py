import ir


def test_move_out_first():
    with ir.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "inout", "cpu"),
    ]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(i == 0):
                y[i] = 0
            y[i] = y[i] + x[i]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "inout", "cpu"),
    ]) as (x, y):
        y[0] = 0
        with ir.For("i", 0, 4) as i:
            y[i] = y[i] + x[i]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_move_out_last():
    with ir.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "inout", "cpu"),
    ]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = y[i] + x[i]
            with ir.If(i == 3):
                y[i] = 0
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "inout", "cpu"),
    ]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = y[i] + x[i]
        y[3] = 0
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)
