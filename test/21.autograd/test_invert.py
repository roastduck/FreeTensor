import freetensor as ft


def test_free_from_recomp():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("w", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, w, y):
        with ft.VarDef("s", (), "float32", "cache", "cpu") as s:
            s[...] = 0
            with ft.For("i", 0, 4) as i:
                s[...] += w[i]
                y[...] += s[...] * x[i]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x", "w"], ["y"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("x", (4,), "float32", "input", "cpu"),
        ("dx", (4,), "float32", "output", "cpu"),
        ("w", (4,), "float32", "input", "cpu"),
        ("dw", (4,), "float32", "output", "cpu"),
        ("dy", (), "float32", "inout", "cpu"),
    ]) as (x, dx, w, dw, dy):
        with ft.VarDef("ds", (), "float32", "cache", "cpu") as ds:
            ds[...] = 0
            with ft.VarDef("s", (), "float32", "cache", "cpu") as s:
                s[...] = 0
                with ft.For("i", 0, 4) as i:
                    s[...] += w[i]
                with ft.For("i", 3, -1, -1) as i:
                    ds[...] += dy[...] * x[i]
                    dx[i] = dy[...] * s[...]
                    dw[i] = ds[...]
                    with ft.If(i > 0):
                        s[...] += -1 * w[i]
    std = ft.pop_ast()

    assert std.match(ast)
