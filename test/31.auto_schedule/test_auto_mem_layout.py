import freetensor as ft
import pytest


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_basic():
    with ft.VarDef([("x", (1000, 1000), "float32", "input", "cpu"),
                    ("y", (1000,), "float32", "output", "cpu")]) as (x, y):
        ft.MarkLabel("Vt")
        with ft.VarDef("t", (1000, 2), "float32", "cache", "cpu") as t:
            with ft.For("i", 0, 1000) as i:
                t[i, 0] = 0
                t[i, 1] = 0
                with ft.For("j", 0, 1000) as j:
                    t[i, 0] += ft.sin(x[i, j])
                    t[i, 1] += ft.cos(x[i, j])
            with ft.For("i", 0, 1000) as i:
                y[i] = t[i, 0] + t[i, 1]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_mem_layout(ft.GPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert logs == ['var_split(Vt, 0, 1, 32, -1)', 'var_reorder(Vt, 0, 2, 1)']
