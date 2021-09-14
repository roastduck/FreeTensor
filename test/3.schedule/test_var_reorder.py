import ir
import pytest


def test_basic():
    with ir.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("Dc")
        with ir.VarDef("c", (4, 8), "int32", "cache", "cpu") as c:
            with ir.For("i", 0, 4) as i:
                with ir.For("j", 0, 8) as j:
                    c[i, j] = x[i, j] * 2
            with ir.For("i", 0, 4) as i:
                with ir.For("j", 0, 8) as j:
                    y[i, j] = c[i, j] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.var_reorder("Dc", [1, 0])
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("Dc")
        with ir.VarDef("c", (8, 4), "int32", "cache", "cpu") as c:
            with ir.For("i", 0, 4) as i:
                with ir.For("j", 0, 8) as j:
                    c[j, i] = x[i, j] * 2
            with ir.For("i", 0, 4) as i:
                with ir.For("j", 0, 8) as j:
                    y[i, j] = c[j, i] + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_not_found():
    with ir.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("Dc")
        with ir.VarDef("c", (4, 8), "int32", "cache", "cpu") as c:
            with ir.For("i", 0, 4) as i:
                with ir.For("j", 0, 8) as j:
                    c[i, j] = x[i, j] * 2
            with ir.For("i", 0, 4) as i:
                with ir.For("j", 0, 8) as j:
                    y[i, j] = c[i, j] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.var_reorder("Dx", [1, 0])
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_not_a_permutation():
    with ir.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("Dc")
        with ir.VarDef("c", (4, 8), "int32", "cache", "cpu") as c:
            with ir.For("i", 0, 4) as i:
                with ir.For("j", 0, 8) as j:
                    c[i, j] = x[i, j] * 2
            with ir.For("i", 0, 4) as i:
                with ir.For("j", 0, 8) as j:
                    y[i, j] = c[i, j] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.var_reorder("Dx", [2, 0])
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)
