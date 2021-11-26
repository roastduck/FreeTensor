import ir
import pytest


def test_factor():
    ir.MarkNid("Dy")
    with ir.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 8) as i:
            y[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.var_split("Dy", 0, ir.VarSplitMode.FixedSize, 4)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (2, 4), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 8) as i:
            y[ir.round_towards_0_div(i, 4), i % 4] = i
    std = ir.use_builtin_div(ir.pop_ast())

    assert std.match(ast)


def test_nparts():
    ir.MarkNid("Dy")
    with ir.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 8) as i:
            y[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.var_split("Dy", 0, ir.VarSplitMode.FixedSize, nparts=4)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (4, 2), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 8) as i:
            y[ir.round_towards_0_div(i, 2), i % 2] = i
    std = ir.use_builtin_div(ir.pop_ast())

    assert std.match(ast)


def test_non_divisible():
    ir.MarkNid("Dy")
    with ir.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 10) as i:
            y[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.var_split("Dy", 0, ir.VarSplitMode.FixedSize, 4)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (3, 4), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 10) as i:
            y[ir.round_towards_0_div(i, 4), i % 4] = i
    std = ir.use_builtin_div(ir.pop_ast())

    assert std.match(ast)


def test_not_found():
    ir.MarkNid("Dy")
    with ir.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 8) as i:
            y[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.var_split("Dx", 0, ir.VarSplitMode.FixedSize)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_out_of_range():
    ir.MarkNid("Dy")
    with ir.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 8) as i:
            y[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.var_split("Dy", 1, ir.VarSplitMode.FixedSize)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)
