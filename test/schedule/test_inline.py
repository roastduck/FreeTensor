import ir
import pytest

def test_basic():
    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ir.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 1
    std = ir.pop_ast()

    assert std.match(ast)

def test_no_inline_expr_is_changed():
    with ir.VarDef([
            ("x", (4,), "int32", "inout", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ir.For("i", 0, 4) as i:
                x[i] = 1
            with ir.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.inline("T")
    ast_ = s.ast() # Should not changed
    assert ast_.match(ast)

def test_no_inline_output_var():
    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "output", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ir.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.inline("T")
    ast_ = s.ast() # Should not changed
    assert ast_.match(ast)

