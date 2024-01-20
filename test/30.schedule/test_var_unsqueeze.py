import freetensor as ft
import pytest


def test_basic():
    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        ft.MarkLabel("Dc")
        with ft.VarDef("c", (4, 8), "int32", "cache", "cpu") as c:
            with ft.For("i", 0, 4) as i:
                with ft.For("j", 0, 8) as j:
                    c[i, j] = x[i, j] * 2
            with ft.For("i", 0, 4) as i:
                with ft.For("j", 0, 8) as j:
                    y[i, j] = c[i, j] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.var_unsqueeze("Dc", 1)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=['prop_one_time_use'], verbose=1)

    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        ft.MarkLabel("Dc")
        with ft.VarDef("c", (4, 1, 8), "int32", "cache", "cpu") as c:
            with ft.For("i", 0, 4) as i:
                with ft.For("j", 0, 8) as j:
                    c[i, 0, j] = x[i, j] * 2
            with ft.For("i", 0, 4) as i:
                with ft.For("j", 0, 8) as j:
                    y[i, j] = c[i, 0, j] + 1
    std = ft.pop_ast()


def test_out_of_range():
    ft.MarkLabel("Dy")
    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8) as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.var_unsqueeze("Dy", 2)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)
