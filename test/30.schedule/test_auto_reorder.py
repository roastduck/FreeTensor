import freetensor as ft
import pytest


def test_move_loop_with_dependence_inside():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000, 1000), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 1000, nid="Li") as i:
            with ft.For("j", 1, 1000, nid="Lj") as j:
                y[i, j] = y[i - 1, j] + x[i, j]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_reorder(ft.CPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == ["reorder(Lj, Li)"]
