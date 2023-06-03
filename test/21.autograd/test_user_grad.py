import freetensor as ft
import pytest


def test_basic():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        ft.MarkLabel('Vt')
        with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
            t[...] = x[...] * x[...]
            ft.MarkVersion("t_now", t)
            y[...] = ft.intrinsic("sinf(%)", t[...], ret_type="float32")
            with ft.UserGradStaged(t, y) as (dt, dy):
                dt[...] = dy[...] * ft.intrinsic("cosf(%)",
                                                 ft.load_at_version(
                                                     "t_now", "float32"),
                                                 ret_type="float32")
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], {'Vt'},
                                   user_grads=user_grads)
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("dx", (), "float32", "output", "cpu"),
                    ("dy", (), "float32", "inout", "cpu")]) as (x, dx, dy):
        with ft.VarDef("t", (), "float32>=0", "input", "cpu") as t:
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
            ft.MarkVersion("t_now", t)
            y[...] = ft.intrinsic("sinf(%)", t[...], ret_type="float32")
            with ft.UserGradStaged(t, y) as (dt, dy):
                dt[...] = dy[...] * ft.intrinsic("cosf(%)",
                                                 ft.load_at_version(
                                                     "t_now", "float32"),
                                                 ret_type="float32")
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"],
                                   set(),
                                   user_grads=user_grads)
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
        ft.MarkVersion("x_now", x)
        y[...] = ft.intrinsic("sinf(%)", x[...], ret_type="float32")
        with ft.UserGradStaged(x, y) as (dx, dy):
            dx[...] = dy[...] * ft.intrinsic("cosf(%)",
                                             ft.load_at_version(
                                                 "x_now", "float32"),
                                             ret_type="float32")
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"],
                                   set(),
                                   user_grads=user_grads)
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("dx", (), "float32", "output", "cpu"),
                    ("dy", (), "float32", "inout", "cpu")]) as (x, dx, dy):
        dx[...] = dy[...] * ft.intrinsic("cosf(%)", x[...], ret_type="float32")
    std = ft.pop_ast()

    assert std.match(ast)


def test_use_reduce_sum_for_partial_derivative():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu"),
                    ("z", (), "float32", "output", "cpu")]) as (x, y, z):
        ft.MarkVersion("x_now", x)
        y[...] = ft.intrinsic("sinf(%)", x[...], ret_type="float32")
        with ft.UserGradStaged(x, y) as (dx, dy):
            # Use `+=` here, because we also have `z`
            dx[...] += dy[...] * ft.intrinsic("cosf(%)",
                                              ft.load_at_version(
                                                  "x_now", "float32"),
                                              ret_type="float32")
        z[...] = x[...] * 2
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y", "z"],
                                   set(),
                                   reset_provided_grad=False,
                                   user_grads=user_grads)
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("dx", (), "float32", "output", "cpu"),
                    ("dy", (), "float32", "input", "cpu"),
                    ("dz", (), "float32", "input", "cpu")]) as (x, dx, dy, dz):
        dx[...] = 2 * dz[...] + dy[...] * ft.intrinsic(
            "cosf(%)", x[...], ret_type="float32")
    std = ft.pop_ast()

    assert std.match(ast)


def test_use_reduce_sum_for_non_partial_derivative():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        ft.MarkVersion("x_now", x)
        y[...] = ft.intrinsic("sinf(%)", x[...], ret_type="float32")
        with ft.UserGradStaged(x, y) as (dx, dy):
            # Although `y` is the only function of `x`, we should still allow using
            # `+=` here
            dx[...] += dy[...] * ft.intrinsic("cosf(%)",
                                              ft.load_at_version(
                                                  "x_now", "float32"),
                                              ret_type="float32")
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"],
                                   set(),
                                   user_grads=user_grads)
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
                ft.MarkVersion("sin_now", sin)
                ft.MarkVersion("cos_now", cos)
                y[...] = sin[...] * sin[...] - cos[...] * cos[...]
        with ft.UserGradStaged(x, y) as (dx, dy):
            dx[...] = 4 * dy[...] * ft.load_at_version(
                'cos_now', 'float32') * ft.load_at_version(
                    'sin_now', 'float32')
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], {'Vsin', 'Vcos'},
                                   user_grads=user_grads)
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


def test_user_grad_on_range_crossing_def():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("z", (), "float32", "output", "cpu")]) as (x, z):
        # (sin^2 x - cos^2 x)' = 4 * cos x * sin x
        ft.MarkLabel('Vsin')
        with ft.VarDef("sin", (), "float32", "cache", "cpu") as sin:
            rng = ft.StmtRange()
            rng.__enter__()
            sin[...] = ft.intrinsic("sinf(%)", x[...], ret_type="float32")
            ft.MarkLabel('Vcos')
            with ft.VarDef("cos", (), "float32", "cache", "cpu") as cos:
                cos[...] = ft.intrinsic("cosf(%)", x[...], ret_type="float32")
                ft.MarkVersion("sin_now", sin)
                ft.MarkVersion("cos_now", cos)
                with ft.VarDef("y", (), "float32", "cache", "cpu") as y:
                    y[...] = sin[...] * sin[...] - cos[...] * cos[...]
                    rng.__exit__(None, None, None)
                    with ft.UserGradStaged(x, y, stmt_range=rng) as (dx, dy):
                        dx[...] = 4 * dy[...] * ft.load_at_version(
                            'cos_now', 'float32') * ft.load_at_version(
                                'sin_now', 'float32')
                    z[...] = 2 * y[...]
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["z"], {'Vsin', 'Vcos'},
                                   reset_provided_grad=False,
                                   user_grads=user_grads)
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("dx", (), "float32", "output", "cpu"),
                    ("dz", (), "float32", "input", "cpu"),
                    ("sin", (), "float32", "input", "cpu"),
                    ("cos", (), "float32", "input", "cpu")]) as (dx, dz, sin,
                                                                 cos):
        dx[...] = 8 * dz[...] * cos[...] * sin[...]
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
                ft.MarkVersion("sin_now", sin)
                ft.MarkVersion("cos_now", cos)
                y[...] = sin[...] * sin[...] - cos[...] * cos[...]
        with ft.UserGradStaged(x, y) as (dx, dy):
            dx[...] = 4 * dy[...] * ft.load_at_version(
                'cos_now', 'float32') * ft.load_at_version(
                    'sin_now', 'float32')
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"],
                                   set(),
                                   user_grads=user_grads)
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
                ft.MarkVersion("t_now", t)
                y[i] = ft.intrinsic("sinf(%)", t[...], ret_type="float32")
                with ft.UserGradStaged(t, y) as (dt, dy):
                    dt[...] = dy[i] * ft.intrinsic("cosf(%)",
                                                   ft.load_at_version(
                                                       "t_now", "float32"),
                                                   ret_type="float32")
    ast, user_grads = ft.pop_ast_and_user_grads()

    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], {'Vt'},
                                   user_grads=user_grads)
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("dx", (4,), "float32", "output", "cpu"),
                    ("dy", (4,), "float32", "inout", "cpu")]) as (x, dx, dy):
        with ft.VarDef("t", (4,), "float32>=0", "input", "cpu") as t:
            with ft.For("i", 3, -1, -1) as i:
                dx[i] = 2 * (dy[i] * ft.intrinsic(
                    "cosf(%)", t[i], ret_type="float32") * x[i])
    std = ft.pop_ast()

    assert std.match(ast)


def test_frontend():

    @ft.transform(verbose=2)
    def func(x: ft.Var[(4,), "float32"]):
        # (sin^2 x - cos^2 x)' = 4 * cos x * sin x
        with ft.StmtRange() as rng:
            sin = ft.unary_op(
                lambda item: ft.intrinsic("sinf(%)", item, ret_type="float32"),
                x)
            cos = ft.unary_op(
                lambda item: ft.intrinsic("cosf(%)", item, ret_type="float32"),
                x)
            sin_now = ft.push_for_backward(sin)
            cos_now = ft.push_for_backward(cos)
            y = sin * sin - cos * cos
        with ft.UserGrad(x, y, stmt_range=rng) as (dx, dy):
            dx[...] = 4 * dy * cos_now * sin_now
        z = y * 2
        return z

    _, bwd, _, _ = ft.grad(func, ["x"], [ft.Return()],
                           ft.GradTapeMode.All,
                           tape_in_closure=True,
                           reset_provided_grad=False,
                           user_grads=func.user_grads)
    print(bwd)
    bwd = ft.lower(bwd, verbose=1)

    with ft.VarDef([("dx", (4,), "float32", "output", "cpu"),
                    ("sin", (4,), "float32", "input", "cpu"),
                    ("cos", (4,), "float32", "input", "cpu"),
                    ("dz", (4,), "float32", "input", "cpu")]) as (dx, sin, cos,
                                                                  dz):
        with ft.For("i", 0, 4) as i:
            dx[i] = 8 * dz[i] * cos[i] * sin[i]
    std = ft.pop_ast()

    assert std.match(bwd.body)


def test_stmt_range_robustness():
    # If we call StmtRange on some volatile statements, it must still work

    @ft.transform(verbose=2)
    def func(x: ft.Var[(4,), "float32"]):
        # (sin^2 x - cos^2 x)' = 4 * cos x * sin x
        with ft.StmtRange() as rng:
            # Useless statement that will be removed soon
            with ft.NamedScope():
                pass

            sin = ft.unary_op(
                lambda item: ft.intrinsic("sinf(%)", item, ret_type="float32"),
                x)
            cos = ft.unary_op(
                lambda item: ft.intrinsic("cosf(%)", item, ret_type="float32"),
                x)
            sin_now = ft.push_for_backward(sin)
            cos_now = ft.push_for_backward(cos)
            y = sin * sin - cos * cos

            # Useless statement that will be removed soon
            with ft.NamedScope():
                pass
        with ft.UserGrad(x, y, stmt_range=rng) as (dx, dy):
            dx[...] = 4 * dy * cos_now * sin_now
        z = y * 2
        return z

    _, bwd, _, _ = ft.grad(func, ["x"], [ft.Return()],
                           ft.GradTapeMode.All,
                           tape_in_closure=True,
                           reset_provided_grad=False,
                           user_grads=func.user_grads)
    print(bwd)
    bwd = ft.lower(bwd, verbose=1)

    with ft.VarDef([("dx", (4,), "float32", "output", "cpu"),
                    ("sin", (4,), "float32", "input", "cpu"),
                    ("cos", (4,), "float32", "input", "cpu"),
                    ("dz", (4,), "float32", "input", "cpu")]) as (dx, sin, cos,
                                                                  dz):
        with ft.For("i", 0, 4) as i:
            dx[i] = 8 * dz[i] * cos[i] * sin[i]
    std = ft.pop_ast()

    assert std.match(bwd.body)


def test_same_mark_version_name_in_different_call_site():

    @ft.inline
    def callee(x):
        # (sin^2 x - cos^2 x)' = 4 * cos x * sin x
        with ft.StmtRange() as rng:
            sin = ft.unary_op(
                lambda item: ft.intrinsic("sinf(%)", item, ret_type="float32"),
                x)
            cos = ft.unary_op(
                lambda item: ft.intrinsic("cosf(%)", item, ret_type="float32"),
                x)
            sin_now = ft.push_for_backward(sin)
            cos_now = ft.push_for_backward(cos)
            y = sin * sin - cos * cos
        with ft.UserGrad(x, y, stmt_range=rng) as (dx, dy):
            dx[...] = 4 * dy * cos_now * sin_now
        return y

    @ft.transform(verbose=2)
    def func(x1: ft.Var[(4,), "float32"], x2: ft.Var[(4,), "float32"]):
        return callee(x1) + callee(x2)

    print(func.user_grads)
    _, bwd, _, _ = ft.grad(func, ["x1", "x2"], [ft.Return()],
                           ft.GradTapeMode.All,
                           tape_in_closure=True,
                           reset_provided_grad=False,
                           user_grads=func.user_grads)
    print(bwd)
    bwd = ft.lower(bwd, verbose=1)

    with ft.VarDef([("dx1", (4,), "float32", "output", "cpu"),
                    ("dx2", (4,), "float32", "output", "cpu"),
                    ("sin1", (4,), "float32", "input", "cpu"),
                    ("cos1", (4,), "float32", "input", "cpu"),
                    ("sin2", (4,), "float32", "input", "cpu"),
                    ("cos2", (4,), "float32", "input", "cpu"),
                    ("dz", (4,), "float32", "input", "cpu")
                   ]) as (dx1, dx2, sin1, cos1, sin2, cos2, dz):
        with ft.For("i", 0, 4) as i:
            dx2[i] = 4 * dz[i] * cos2[i] * sin2[i]
        with ft.For("i", 0, 4) as i:
            dx1[i] = 4 * dz[i] * cos1[i] * sin1[i]
    std = ft.pop_ast()

    assert std.match(bwd.body)


def test_error_missing_user_grad():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        ft.MarkLabel('Vt')
        with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
            t[...] = x[...] * x[...]
            y[...] = ft.intrinsic("sinf(%)", t[...], ret_type="float32")
    ast, user_grads = ft.pop_ast_and_user_grads()

    with pytest.raises(ft.InvalidAutoGrad):
        ft.grad_body(ast, ["x"], ["y"], {'Vt'}, user_grads=user_grads)
