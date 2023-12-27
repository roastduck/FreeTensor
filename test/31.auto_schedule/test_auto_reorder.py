import freetensor as ft
import pytest


def test_move_loop_with_dependence_inside():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000, 1000), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 1000, label="Li") as i:
            with ft.For("j", 1, 1000, label="Lj") as j:
                y[i, j] = y[i - 1, j] + x[i, j]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_reorder(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert logs == ["reorder(Lj, Li, move_out_imperfect)"]


def test_true_dep_is_more_important_than_reduction():
    with ft.VarDef([("x", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000, 1000), "int32", "inout", "cpu"),
                    ("z", (1000,), "int32", "inout", "cpu")]) as (x, y, z):
        with ft.For("i", 0, 1000, label="Li") as i:
            with ft.For("j", 1, 1000, label="Lj") as j:
                y[i, j] = y[i - 1, j] + x[i, j]
                z[i] += x[i, j]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_reorder(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert logs == ["reorder(Lj, Li, move_out_imperfect)"]


def test_no_reorder_inner_if_outer_has_enough_degree_of_parallelism():
    with ft.VarDef([("x", (1000, 1000, 4), "int32", "input", "cpu"),
                    ("y", (1000, 1000), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 1000, label="Li") as i:
            with ft.For("j", 1, 1000, label="Lj") as j:
                with ft.For("k", 1, 4, label="Lk") as k:
                    y[i, k] += x[i, j, k]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_reorder(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert logs == []
