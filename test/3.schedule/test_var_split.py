import freetensor as ft
import pytest


def test_factor():
    ft.MarkNid("Dy")
    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8) as i:
            y[i] = i
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.var_split("Dy", 0, ft.VarSplitMode.FixedSize, 4)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef("y", (2, 4), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8) as i:
            y[ft.round_towards_0_div(i, 4), i % 4] = i
    std = ft.use_builtin_div(ft.pop_ast())

    assert std.match(ast)


def test_nparts():
    ft.MarkNid("Dy")
    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8) as i:
            y[i] = i
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.var_split("Dy", 0, ft.VarSplitMode.FixedSize, nparts=4)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef("y", (4, 2), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8) as i:
            y[ft.round_towards_0_div(i, 2), i % 2] = i
    std = ft.use_builtin_div(ft.pop_ast())

    assert std.match(ast)


def test_non_divisible():
    ft.MarkNid("Dy")
    with ft.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 10) as i:
            y[i] = i
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.var_split("Dy", 0, ft.VarSplitMode.FixedSize, 4)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef("y", (3, 4), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 10) as i:
            y[ft.round_towards_0_div(i, 4), i % 4] = i
    std = ft.use_builtin_div(ft.pop_ast())

    assert std.match(ast)


def test_not_found():
    ft.MarkNid("Dy")
    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8) as i:
            y[i] = i
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.var_split("Dx", 0, ft.VarSplitMode.FixedSize)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_out_of_range():
    ft.MarkNid("Dy")
    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8) as i:
            y[i] = i
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.var_split("Dy", 1, ft.VarSplitMode.FixedSize)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)
