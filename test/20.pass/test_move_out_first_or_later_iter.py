import freetensor as ft


def test_move_out_first():
    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "inout", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(i == 0):
                y[i] = 0
            y[i] += x[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "inout", "cpu"),
    ]) as (x, y):
        y[0] = 0
        with ft.For("i", 0, 4) as i:
            y[i] += x[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_move_out_last():
    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "inout", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] += x[i]
            with ft.If(i == 3):
                y[i] = 0
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "inout", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] += x[i]
        y[3] = 0
    std = ft.pop_ast()

    assert std.match(ast)
