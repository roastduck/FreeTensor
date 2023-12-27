import freetensor as ft


def test_basic():
    with ft.VarDef([("x", (1000,), "float32", "input", "cpu"),
                    ("y", (1000,), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 1000) as i:
            ft.MarkLabel("Vt")
            with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
                t[...] = x[i]
                y[i] = ft.sin(t[...]) * ft.cos(t[...])

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_inline(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert logs == ["inline(Vt)"]


def test_order():
    with ft.VarDef([("x", (1000, 3), "float32", "input", "cpu"),
                    ("y", (1000,), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 1000) as i:
            ft.MarkLabel("Vt")
            with ft.VarDef("t", (3,), "float32", "cache", "cpu") as t:
                t[0] = x[i, 0] + x[i, 1]
                t[1] = x[i, 1] + x[i, 2]
                t[2] = x[i, 2] + x[i, 0]
                ft.MarkLabel("Vu")
                with ft.VarDef("u", (), "float32", "cache", "cpu") as u:
                    u[...] = t[0] + t[1] + t[2]
                    y[i] = ft.sin(u[...]) * ft.cos(u[...])

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_inline(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert logs == ["inline(Vu)", "inline(Vt)"]
