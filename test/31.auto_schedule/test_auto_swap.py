import freetensor as ft
import pytest


def test_basic():
    with ft.VarDef([("x1", (1000,), "int32", "input", "cpu"),
                    ("x2", (1000,), "int32", "input", "cpu"),
                    ("y", (1001,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ft.For("i", 0, 1001, label="L0") as i:
            y[i] = 0
        with ft.For("i", 0, 1000, label="L1") as i:
            ft.MarkLabel("S1")
            y[i] += x1[i]  # `+=` here, always happens afterhand
            ft.MarkLabel("S2")
            y[i + 1] = x2[i]  # `=` here, always happens beforehand

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_swap(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert logs == ["swap(S2, S1)"]
