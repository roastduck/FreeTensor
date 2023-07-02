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


def test_true_dep_is_more_inner_than_reduction():
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


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_cuda_loop_carried_reduction_is_more_inner_than_random_reduction():
    # If we reorder `i` inside and parallelize `j`, the reduction on `i` can be
    # done efficiently in serial (a pull-style reduction).
    #
    # If we don't reorder and parallelize `i`, the reduction on `j` still incurs
    # atomic operation (a push-style reduction).

    with ft.VarDef([
        ("x", (1000, 1000), "int32", "input", "cpu"),
        ("idx", (1000,), "int32", "input", "cpu"),
        ("y", (1000,), "int32", "inout", "cpu"),
    ]) as (x, idx, y):
        with ft.For("i", 0, 1000, label="Li") as i:
            with ft.For("j", 1, 1000, label="Lj") as j:
                y[idx[j]] += x[i, j]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_reorder(ft.GPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert logs == ["reorder(Lj, Li, move_out_imperfect)"]
