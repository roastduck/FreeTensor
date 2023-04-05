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
                    with ft.If(i > 0):
                        s[...] += -1 * w[i]
                    dw[i] = ds[...]
    std = ft.pop_ast()

    assert std.match(ast)


def test_free_from_tape():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("w", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, w, y):
        with ft.VarDef("s", (), "float32", "cache", "cpu") as s:
            s[...] = 0
            with ft.For("i", 0, 4) as i:
                s[...] += w[i]
                y[...] += s[...] * x[i]
    ast = ft.pop_ast(verbose=True)
    fwd, bwd, _, _, _ = ft.grad_body(ast, ["x", "w"], ["y"],
                                     ft.GradTapeMode.All)
    fwd = ft.lower(fwd, verbose=1)
    bwd = ft.lower(bwd, verbose=1)

    with ft.VarDef([
        ("x", (4,), "float32", "input", "cpu"),
        ("dx", (4,), "float32", "output", "cpu"),
        ("w", (4,), "float32", "input", "cpu"),
        ("dw", (4,), "float32", "output", "cpu"),
        ("dy", (), "float32", "inout", "cpu"),
    ]) as (x, dx, w, dw, dy):
        with ft.VarDef("s", (), "float32", "input-mutable", "cpu") as s:
            with ft.VarDef("ds", (), "float32", "cache", "cpu") as ds:
                ds[...] = 0
                with ft.For("i", 3, -1, -1) as i:
                    ds[...] += dy[...] * x[i]
                    dx[i] = dy[...] * s[...]
                    with ft.If(i > 0):
                        s[...] += -1 * w[i]
                    dw[i] = ds[...]
    std = ft.pop_ast()

    assert std.match(bwd)


def test_reduce_mul_gt0():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("w", (4,), "float32>0", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, w, y):
        with ft.VarDef("s", (), "float32", "cache", "cpu") as s:
            s[...] = 1
            with ft.For("i", 0, 4) as i:
                s[...] *= w[i]
                y[...] += s[...] * x[i]
    ast = ft.pop_ast(verbose=True)
    fwd, bwd, _, _, _ = ft.grad_body(ast, ["x"], ["y"], ft.GradTapeMode.All)
    fwd = ft.lower(fwd, verbose=1)
    bwd = ft.lower(bwd, verbose=1)

    with ft.VarDef([
        ("dx", (4,), "float32", "output", "cpu"),
        ("w", (4,), "float32>0", "input", "cpu"),
        ("dy", (), "float32", "inout", "cpu"),
    ]) as (dx, w, dy):
        with ft.VarDef("s", (), "float32>0", "input-mutable", "cpu") as s:
            with ft.For("i", 3, -1, -1) as i:
                dx[i] = dy[...] * s[...]
                with ft.If(i > 0):
                    s[...] *= 1 / w[i]
    std = ft.pop_ast()

    assert std.match(bwd)


def test_reduce_mul_may_eq0_no_invert():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("w", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, w, y):
        with ft.VarDef("s", (), "float32", "cache", "cpu") as s:
            s[...] = 1
            with ft.For("i", 0, 4) as i:
                s[...] *= w[i]
                y[...] += s[...] * x[i]
    ast = ft.pop_ast(verbose=True)
    fwd, bwd, _, _, _ = ft.grad_body(ast, ["x"], ["y"], ft.GradTapeMode.All)
    fwd = ft.lower(fwd, verbose=1)
    bwd = ft.lower(bwd, verbose=1)

    with ft.VarDef([
        ("dx", (4,), "float32", "output", "cpu"),
        ("dy", (), "float32", "inout", "cpu"),
    ]) as (dx, dy):
        with ft.VarDef("s", (4,), "float32", "input", "cpu") as s:
            with ft.For("i", 3, -1, -1) as i:
                dx[i] = dy[...] * s[i]
    std = ft.pop_ast()

    assert std.match(bwd)


def test_reduce_mul_invert_then_grad():
    with ft.VarDef([("x", (4,), "float32>0", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[...] = 1
        with ft.For("i", 0, 4) as i:
            y[...] *= x[i]
    ast = ft.pop_ast(verbose=True)
    fwd, bwd, _, _, _ = ft.grad_body(ast, ["x"], ["y"], ft.GradTapeMode.All)
    fwd = ft.lower(fwd, verbose=1)
    bwd = ft.lower(bwd, verbose=1)

    # NOTE: (Non-goal) This can be further optimized if we have special gradient
    # rule for cumulative product
    with ft.VarDef([
        ("x", (4,), "float32>0", "input", "cpu"),
        ("dx", (4,), "float32", "output", "cpu"),
        ("y", (), "float32>0", "input-mutable", "cpu"),
        ("dy", (), "float32", "inout", "cpu"),
    ]) as (x, dx, y, dy):
        with ft.For("i", 3, -1, -1) as i:
            # INVERT FIRST
            y[...] *= 1 / x[i]
            # AND THEN COMPUTE GRADIENT
            with ft.VarDef("dy_old", (), "float32", "cache", "cpu") as dy_old:
                dy_old[...] = dy[...]
                dy[...] = dy_old[...] * x[i]
                dx[i] = dy_old[...] * y[...]
        dy[...] = 0
    std = ft.pop_ast()

    assert std.match(bwd)
