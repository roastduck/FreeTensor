import freetensor as ft


def test_basic():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        ft.MarkLabel('Vt')
        with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
            t[...] = x[...] * x[...]
            ft.MarkVersion("t0", t)
            y[...] = ft.intrinsic("sinf(%)", t[...], ret_type="float32")
            with ft.UserGradForPrevStmt(t, y) as (dt, dy):
                dt[...] = dy[...] * ft.intrinsic(
                    "cosf(%)", ft.load_at_version("t0"), ret_type="float32")
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], {'Vt'}, user_grads)
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("dx", (), "float32", "output", "cpu"),
                    ("dy", (), "float32", "inout", "cpu")]) as (x, dx, dy):
        with ft.VarDef("t", (), "float32", "input", "cpu") as t:
            dx[...] = 2 * (dy[...] * ft.intrinsic(
                "cosf(%)", t[...], ret_type="float32") * x[...])
    std = ft.pop_ast()

    assert std.match(ast)


def test_marked_version_is_recomputed():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        ft.MarkLabel('Vt')
        with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
            t[...] = x[...] * x[...]
            ft.MarkVersion("t0", t)
            y[...] = ft.intrinsic("sinf(%)", t[...], ret_type="float32")
            with ft.UserGradForPrevStmt(t, y) as (dt, dy):
                dt[...] = dy[...] * ft.intrinsic(
                    "cosf(%)", ft.load_at_version("t0"), ret_type="float32")
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], set(), user_grads)
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("dx", (), "float32", "output", "cpu"),
                    ("dy", (), "float32", "inout", "cpu")]) as (x, dx, dy):
        dx[...] = 2 * (dy[...] * ft.intrinsic(
            "cosf(%)", ft.square(x[...]), ret_type="float32") * x[...])
    std = ft.pop_ast()

    assert std.match(ast)


def test_mark_version_on_input():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        ft.MarkVersion("x0", x)
        y[...] = ft.intrinsic("sinf(%)", x[...], ret_type="float32")
        with ft.UserGradForPrevStmt(x, y) as (dx, dy):
            dx[...] = dy[...] * ft.intrinsic(
                "cosf(%)", ft.load_at_version("x0"), ret_type="float32")
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], set(), user_grads)
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("dx", (), "float32", "output", "cpu"),
                    ("dy", (), "float32", "inout", "cpu")]) as (x, dx, dy):
        dx[...] = dy[...] * ft.intrinsic("cosf(%)", x[...], ret_type="float32")
    std = ft.pop_ast()

    assert std.match(ast)


def test_user_grad_on_scope():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        # (sin^2 x - cos^2 x)' = 4 * cos x * sin x
        ft.MarkLabel('Vsin')
        with ft.VarDef("sin", (), "float32", "cache", "cpu") as sin:
            ft.MarkLabel('Vcos')
            with ft.VarDef("cos", (), "float32", "cache", "cpu") as cos:
                sin[...] = ft.intrinsic("sinf(%)", x[...], ret_type="float32")
                cos[...] = ft.intrinsic("cosf(%)", x[...], ret_type="float32")
                ft.MarkVersion("sin0", sin)
                ft.MarkVersion("cos0", cos)
                y[...] = sin[...] * sin[...] - cos[...] * cos[...]
        with ft.UserGradForPrevStmt(x, y) as (dx, dy):
            dx[...] = 4 * dy[...] * ft.load_at_version(
                'cos0') * ft.load_at_version('sin0')
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], {'Vsin', 'Vcos'},
                                   user_grads)
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("dx", (), "float32", "output", "cpu"),
                    ("dy", (), "float32", "inout", "cpu"),
                    ("sin", (), "float32", "input", "cpu"),
                    ("cos", (), "float32", "input", "cpu")]) as (dx, dy, sin,
                                                                 cos):
        dx[...] = 4 * dy[...] * cos[...] * sin[...]
    std = ft.pop_ast()

    assert std.match(ast)


def test_user_grad_on_scope_with_load_at_version_recomputed():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        # (sin^2 x - cos^2 x)' = 4 * cos x * sin x
        with ft.VarDef("sin", (), "float32", "cache", "cpu") as sin:
            with ft.VarDef("cos", (), "float32", "cache", "cpu") as cos:
                sin[...] = ft.intrinsic("sinf(%)", x[...], ret_type="float32")
                cos[...] = ft.intrinsic("cosf(%)", x[...], ret_type="float32")
                ft.MarkVersion("sin0", sin)
                ft.MarkVersion("cos0", cos)
                y[...] = sin[...] * sin[...] - cos[...] * cos[...]
        with ft.UserGradForPrevStmt(x, y) as (dx, dy):
            dx[...] = 4 * dy[...] * ft.load_at_version(
                'cos0') * ft.load_at_version('sin0')
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], set(), user_grads)
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("dx", (), "float32", "output", "cpu"),
                    ("dy", (), "float32", "inout", "cpu")]) as (x, dx, dy):
        dx[...] = 4 * dy[...] * ft.intrinsic(
            "cosf(%)", x[...], ret_type="float32") * ft.intrinsic(
                "sinf(%)", x[...], ret_type="float32")
    std = ft.pop_ast()

    assert std.match(ast)


def test_mark_from_multiple_versions():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            ft.MarkLabel('Vt')
            with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
                t[...] = x[i] * x[i]
                ft.MarkVersion("t0", t)
                y[i] = ft.intrinsic("sinf(%)", t[...], ret_type="float32")
                with ft.UserGradForPrevStmt(t, y) as (dt, dy):
                    dt[...] = dy[i] * ft.intrinsic(
                        "cosf(%)", ft.load_at_version("t0"), ret_type="float32")
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], {'Vt'}, user_grads)
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("dx", (4,), "float32", "output", "cpu"),
                    ("dy", (4,), "float32", "inout", "cpu")]) as (x, dx, dy):
        with ft.For("i", 3, -1, -1) as i:
            with ft.VarDef("t", (4,), "float32", "input", "cpu") as t:
                dx[i] = 2 * (dy[i] * ft.intrinsic(
                    "cosf(%)", t[i], ret_type="float32") * x[i])
    std = ft.pop_ast()

    assert std.match(ast)
