import ir
import pytest


def test_basic():
    ir.MarkNid("Dy")
    with ir.VarDef("y", (7, 8), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 7) as i:
            with ir.For("j", 0, 8) as j:
                y[i, j] = i + j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.var_merge("Dy", 0)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (56,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 7) as i:
            with ir.For("j", 0, 8) as j:
                y[i * 8 + j] = i + j
    std = ir.use_builtin_div(ir.pop_ast())

    assert std.match(ast)


def test_not_found():
    ir.MarkNid("Dy")
    with ir.VarDef("y", (7, 8), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 7) as i:
            with ir.For("j", 0, 8) as j:
                y[i, j] = i + j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.var_merge("Dx", 0)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_out_of_range():
    ir.MarkNid("Dy")
    with ir.VarDef("y", (7, 8), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 7) as i:
            with ir.For("j", 0, 8) as j:
                y[i, j] = i + j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.var_merge("Dy", 1)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)
