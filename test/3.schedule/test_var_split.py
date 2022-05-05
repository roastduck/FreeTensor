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
            y[i // 4, i % 4] = i
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
            y[i // 2, i % 2] = i
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
            y[i // 4, i % 4] = i
    std = ft.use_builtin_div(ft.pop_ast())

    assert std.match(ast)


def test_non_divisible_when_caching():
    ft.MarkNid("Dx")
    with ft.VarDef("x", (10,), "int32", "input", "cpu") as x:
        with ft.VarDef([("y", (10,), "int32", "output", "cpu"),
                        ("z", (10,), "int32", "output", "cpu")]) as (y, z):
            with ft.For("i", 0, 10, nid="Li") as i:
                y[i] = x[i] + 1
                z[i] = x[i] + 2
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.var_split("Dx", 0, ft.VarSplitMode.FixedSize, 4)
    s.cache("Li", "x", "cpu")
    ast = s.ast()
    print(ast)

    with ft.VarDef([("x", (3, 4), "int32", "input", "cpu"),
                    ("y", (10,), "int32", "output", "cpu"),
                    ("z", (10,), "int32", "output", "cpu")]) as (x, y, z):
        with ft.VarDef("t", (3, 4), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 3) as i:
                with ft.For("j", 0, 4) as j:
                    with ft.If(i * 4 + j < 10):
                        t[i, j] = x[i, j]
            with ft.For("k", 0, 10, nid="Li") as k:
                y[k] = t[k // 4, k % 4] + 1
                z[k] = t[k // 4, k % 4] + 2
    std = ft.pop_ast()

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
