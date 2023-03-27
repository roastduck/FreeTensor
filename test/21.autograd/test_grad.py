import pytest
import freetensor as ft


def test_basic():
    with ft.VarDef([
        ("x1", (), "float32", "input", "cpu"),
        ("x2", (), "float32", "input", "cpu"),
        ("x3", (), "float32", "input", "cpu"),
        ("y", (), "float32", "output", "cpu"),
    ]) as (x1, x2, x3, y):
        ft.MarkLabel("S0")
        y[()] = (x1[()] + x2[()]) * x3[()]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x1", "x2", "x3"], ["y"], set())
    print(ast)
    assert len(ft.find_all_stmt(ast, "$grad{S0}")) > 0
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("d_x1", (), "float32", "output", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("d_x2", (), "float32", "output", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("d_x3", (), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, d_y):
        d_x1[()] = d_y[()] * x3[()]
        d_x2[()] = d_y[()] * x3[()]
        d_x3[()] = d_y[()] * (x1[()] + x2[()])
    std = ft.pop_ast()

    assert std.match(ast)


def test_partial_gradient():
    with ft.VarDef([
        ("x1", (), "float32", "input", "cpu"),
        ("x2", (), "float32", "input", "cpu"),
        ("x3", (), "float32", "input", "cpu"),
        ("y", (), "float32", "output", "cpu"),
    ]) as (x1, x2, x3, y):
        ft.MarkLabel("S0")
        y[()] = (x1[()] + x2[()]) * x3[()]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x1"], ["y"], set())
    print(ast)
    assert len(ft.find_all_stmt(ast, "$grad{S0}")) > 0
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("d_x1", (), "float32", "output", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (d_x1, x3, d_y):
        d_x1[()] = d_y[()] * x3[()]
    std = ft.pop_ast()

    assert std.match(ast)


def test_branching_exprs():
    with ft.VarDef([
        ("x1", (), "float32", "input", "cpu"),
        ("x2", (), "float32", "input", "cpu"),
        ("y1", (), "float32", "output", "cpu"),
        ("y2", (), "float32", "output", "cpu"),
        ("y3", (), "float32", "output", "cpu"),
    ]) as (x1, x2, y1, y2, y3):
        y1[()] = ft.min(x1[()], x2[()])
        y2[()] = ft.max(x1[()], x2[()])
        y3[()] = ft.if_then_else(x1[()] > 0, x1[()], x2[()])
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x1", "x2"], ["y1", "y2", "y3"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("d_x1", (), "float32", "output", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("d_x2", (), "float32", "output", "cpu"),
                    ("d_y1", (), "float32", "inout", "cpu"),
                    ("d_y2", (), "float32", "inout", "cpu"),
                    ("d_y3", (), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, d_y1, d_y2, d_y3):
        d_x1[()] = ft.if_then_else(x1[()] > 0, d_y3[()], 0) + ft.if_then_else(
            x1[()] >= x2[()], d_y2[()], 0) + ft.if_then_else(
                x1[()] <= x2[()], d_y1[()], 0)
        d_x2[()] = ft.if_then_else(x1[()] <= 0, d_y3[()], 0) + ft.if_then_else(
            x2[()] > x1[()], d_y2[()], 0) + ft.if_then_else(
                x2[()] < x1[()], d_y1[()], 0)
    std = ft.pop_ast()

    assert std.match(ast)


def test_math_funcs():
    with ft.VarDef([
        ("x", (), "float32", "input", "cpu"),
        ("y1", (), "float32", "output", "cpu"),
        ("y2", (), "float32", "output", "cpu"),
        ("y3", (), "float32", "output", "cpu"),
    ]) as (x, y1, y2, y3):
        y1[()] = ft.sqrt(x[()])
        y2[()] = ft.exp(x[()])
        y3[()] = ft.square(x[()])
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y1", "y2", "y3"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("d_x", (), "float32", "output", "cpu"),
                    ("d_y1", (), "float32", "inout", "cpu"),
                    ("d_y2", (), "float32", "inout", "cpu"),
                    ("d_y3", (), "float32", "inout", "cpu")]) as (x, d_x, d_y1,
                                                                  d_y2, d_y3):
        d_x[()] = 2 * d_y3[()] * x[()] + d_y2[()] * ft.exp(x[()]) + d_y1[
            ()] * (0.5 / ft.sqrt(x[()]))
    std = ft.pop_ast()

    assert std.match(ast)


def test_use_y_for_grad_when_taped():
    with ft.VarDef("x", (4,), "float32", "input", "cpu") as x:
        ft.MarkLabel("V_y1")
        with ft.VarDef("y1", (4,), "float32", "output", "cpu") as y1:
            ft.MarkLabel("V_y2")
            with ft.VarDef("y2", (4,), "float32", "output", "cpu") as y2:
                with ft.For("i", 0, 4, label="L_i") as i:
                    ft.MarkLabel("V_t")
                    with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
                        t[()] = 2 * x[i]
                        y1[i] = ft.exp(t[()])
                        y2[i] = ft.square(t[()])
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y1", "y2"],
                                   ["V_t", "V_y1", "V_y2"])
    print(ast)
    assert len(ft.find_all_stmt(ast, "$grad{L_i}")) > 0
    assert len(ft.find_all_stmt(ast, "$tape{V_t}")) > 0
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("d_x", (4,), "float32", "output", "cpu"),
        ("y1", (4,), "float32", "input", "cpu"),
        ("d_y1", (4,), "float32", "inout", "cpu"),
        ("d_y2", (4,), "float32", "inout", "cpu"),
    ]) as (d_x, y1, d_y1, d_y2):
        with ft.VarDef("t.tape", (4,), "float32", "input", "cpu") as t:
            with ft.For("i", 3, -1, -1) as i:
                d_x[i] = 2 * (d_y1[i] * y1[i] + 2 * (d_y2[i] * t[i]))
    std = ft.pop_ast()

    assert std.match(ast)


def test_use_taped_y_for_grad():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y1", (4,), "float32", "output", "cpu"),
                    ("y2", (4,), "float32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4, label="L_i") as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
                t[()] = ft.exp(x[i])
                y1[i] = 2 * t[()]
                y2[i] = 3 * t[()]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y1", "y2"], ["V_t"])
    print(ast)
    assert len(ft.find_all_stmt(ast, "$grad{L_i}")) > 0
    assert len(ft.find_all_stmt(ast, "$tape{V_t}")) > 0
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("d_x", (4,), "float32", "output", "cpu"),
        ("d_y1", (4,), "float32", "inout", "cpu"),
        ("d_y2", (4,), "float32", "inout", "cpu"),
    ]) as (d_x, d_y1, d_y2):
        with ft.VarDef("t.tape", (4,), "float32", "input", "cpu") as t:
            with ft.For("i", 3, -1, -1) as i:
                d_x[i] = ((d_y2[i] * 3) + (d_y1[i] * 2)) * t[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_tape_y_to_use_it_for_grad():
    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu")]) as (x1, x2):
        ft.MarkLabel("V_y")
        with ft.VarDef("y", (), "float32", "output", "cpu") as y:
            # `y` must be taped for use in the gradient. We must especially take care with
            # the second statement, as its `y` output isn't used elsewhere, and may not be
            # saved by other means
            y[...] = ft.exp(x1[...])
            y[...] = ft.exp(y[...] + x2[...])

    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x1", "x2"], ["y"], ["V_y"])
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("d_x1", (), "float32", "output", "cpu"),
                    ("d_x2", (), "float32", "output", "cpu"),
                    ("y_tape", (2,), "float32", "input", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (d_x1, d_x2,
                                                                 y_tape, d_y):
        with ft.VarDef("d_y_old", (), "float32", "cache", "cpu") as d_y_old:
            d_y_old[...] = d_y[...]
            d_y[...] = d_y_old[...] * y_tape[1]
            d_x2[...] = d_y_old[...] * y_tape[1]
        d_x1[...] = d_y[...] * y_tape[0]
        d_y[...] = 0
    std = ft.pop_ast()

    assert std.match(ast)


def test_multiple_statements():
    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("y1", (), "float32", "output", "cpu"),
                    ("y2", (), "float32", "output", "cpu")]) as (x1, x2, x3, y1,
                                                                 y2):
        with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
            t[()] = x1[()] + x2[()]
            y1[()] = t[()] * x3[()]
            y2[()] = t[()] + x3[()]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x1", "x2", "x3"], ["y1", "y2"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("d_x1", (), "float32", "output", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("d_x2", (), "float32", "output", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("d_x3", (), "float32", "output", "cpu"),
                    ("d_y1", (), "float32", "inout", "cpu"),
                    ("d_y2", (), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, d_y1, d_y2):
        with ft.VarDef("d_t", (), "float32", "cache", "cpu") as d_t:
            d_t[()] = d_y2[()] + d_y1[()] * x3[()]
            d_x3[()] = d_y2[()] + d_y1[()] * (x1[()] + x2[()])
            d_x1[()] = d_t[()]
            d_x2[()] = d_t[()]
    std = ft.pop_ast()

    assert std.match(ast)


def test_nested_local_def():
    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("y1", (), "float32", "output", "cpu"),
                    ("y2", (), "float32", "output", "cpu")]) as (x1, x2, x3, y1,
                                                                 y2):
        with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
            with ft.VarDef("u", (), "float32", "cache", "cpu") as u:
                u[()] = x1[()] + x2[()]
                t[()] = u[()] * x3[()]
                y1[()] = u[()] * x1[()]
            y2[()] = t[()] * x2[()]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x1", "x2", "x3"], ["y1", "y2"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("d_x1", (), "float32", "output", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("d_x2", (), "float32", "output", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("d_x3", (), "float32", "output", "cpu"),
                    ("d_y1", (), "float32", "inout", "cpu"),
                    ("d_y2", (), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, d_y1, d_y2):
        with ft.VarDef("u", (), "float32", "cache", "cpu") as u:
            u[()] = x1[()] + x2[()]
            with ft.VarDef("d_t", (), "float32", "cache", "cpu") as d_t:
                d_t[()] = d_y2[()] * x2[()]
                with ft.VarDef("d_u", (), "float32", "cache", "cpu") as d_u:
                    d_u[()] = d_y1[()] * x1[()] + d_t[()] * x3[()]
                    d_x3[()] = d_t[()] * u[()]
                    d_x1[()] = d_y1[()] * u[()] + d_u[()]
                    d_x2[()] = d_y2[()] * (u[()] * x3[()]) + d_u[()]
    std = ft.pop_ast()

    assert std.match(ast)


def test_dependent_iterations():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[()] = -y[()] + x[i]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("d_x", (4,), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (d_x, d_y):
        with ft.For("i", 3, -1, -1) as i:
            with ft.VarDef("d_y.old", (), "float32", "cache", "cpu") as d_y_old:
                d_y_old[()] = d_y[()]
                d_y[()] = -1 * d_y_old[()]
                d_x[i] = d_y_old[()]
    std = ft.pop_ast()

    assert std.match(ast)


def test_assign_quick_path():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[()] = x[i]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("d_x", (4,), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (d_x, d_y):
        with ft.For("i", 3, -1, -1) as i:
            d_x[i] = d_y[()]
            d_y[()] = 0
    std = ft.pop_ast()

    assert std.match(ast)


def test_reduce_sum_quick_path():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = 0
        with ft.For("i", 0, 4) as i:
            y[()] += x[i]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("d_x", (4,), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (d_x, d_y):
        with ft.For("i", 3, -1, -1) as i:
            d_x[i] = d_y[()]
        d_y[()] = 0
    std = ft.pop_ast()

    assert std.match(ast)


def test_reduce_sub_quick_path():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = 0
        with ft.For("i", 0, 4) as i:
            y[()] -= x[i]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("d_x", (4,), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (d_x, d_y):
        with ft.For("i", 3, -1, -1) as i:
            d_x[i] = -1 * d_y[()]
        d_y[()] = 0
    std = ft.pop_ast()

    assert std.match(ast)


def test_reduce_min_quick_path():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = float("inf")
        with ft.For("i", 0, 4) as i:
            y[()] = ft.min(y[()], x[i])
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], ft.GradTapeMode.All)
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("d_x", (4,), "float32", "output", "cpu"),
                    ("y", (), "float32", "input", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x, d_x, y,
                                                                 d_y):
        # We can deduce `d_x` from only the final `y`, instead of every 4 versions
        # of `y`
        with ft.For("i", 3, -1, -1) as i:
            d_x[i] = ft.if_then_else(x[i] == y, d_y[()], 0)
            d_y[()] = ft.if_then_else(x[i] == y, 0, d_y[()])
        d_y[()] = 0
    std = ft.pop_ast()

    assert std.match(ast)


def test_reduce_min_quick_path_taped():
    with ft.VarDef([("x", (2, 4), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = 0
        with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
            with ft.For("p", 0, 2) as p:
                t[()] = float("inf")
                with ft.For("i", 0, 4) as i:
                    t[()] = ft.min(t[()], x[p, i])
                y[()] += t[()]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], ft.GradTapeMode.All)
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (2, 4), "float32", "input", "cpu"),
                    ("d_x", (2, 4), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu"),
                    ("t_tape", (2,), "float32", "input", "cpu")
                   ]) as (x, d_x, d_y, t_tape):
        with ft.VarDef("d_t", (), "float32", "cache", "cpu") as d_t:
            with ft.For("p", 1, -1, -1) as p:
                d_t[()] = d_y[()]
                # We need to load a proper versino of `t`
                with ft.For("i", 3, -1, -1) as i:
                    d_x[p, i] = ft.if_then_else(x[p, i] == t_tape[p], d_t[()],
                                                0)
                    d_t[()] = ft.if_then_else(x[p, i] == t_tape[p], 0, d_t[()])
        d_y[()] = 0
    std = ft.pop_ast()

    assert std.match(ast)


def test_no_use_y_for_grad_in_reduce_sum_quick_path():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[()] = 0
        with ft.For("i", 0, 4) as i:
            # We are doing `y += exp(x[i])`, instead of `y = exp(x[i])` here, so
            # don't make `d_x[i] = ? * dy`, because we have nothing equals to `?`.
            # Make `d_x[i] = exp(x[i]) * dy` instead.
            y[()] += ft.exp(x[i])
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("d_x", (4,), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x, d_x, d_y):
        with ft.For("i", 3, -1, -1) as i:
            d_x[i] = d_y[()] * ft.exp(x[i])
        d_y[()] = 0
    std = ft.pop_ast()

    assert std.match(ast)


def test_atypical_loop():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", -2, 2) as i:
            y[()] = -y[()] + x[i + 2]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("d_x", (4,), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (d_x, d_y):
        with ft.For("i", 1, -3, -1) as i:
            with ft.VarDef("d_y.old", (), "float32", "cache", "cpu") as d_y_old:
                d_y_old[()] = d_y[()]
                d_y[()] = -1 * d_y_old[()]
                d_x[i + 2] = d_y_old[()]
    std = ft.pop_ast()

    assert std.match(ast)


def test_nested_loops():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("w0", (4, 4), "float32", "input", "cpu"),
                    ("w1", (4, 4), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x, w0, w1, y):
        with ft.VarDef("t", (4,), "float32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = 0
                with ft.For("j", 0, 4) as j:
                    t[i] += x[j] * w0[i, j]
            with ft.For("i", 0, 4) as i:
                y[i] = 0
                with ft.For("j", 0, 4) as j:
                    y[i] += t[j] * w1[i, j]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x", "w0", "w1"], ["y"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("d_x", (4,), "float32", "output", "cpu"),
                    ("w0", (4, 4), "float32", "input", "cpu"),
                    ("d_w0", (4, 4), "float32", "output", "cpu"),
                    ("w1", (4, 4), "float32", "input", "cpu"),
                    ("d_w1", (4, 4), "float32", "output", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu")
                   ]) as (x, d_x, w0, d_w0, w1, d_w1, d_y):
        with ft.For("i0", 0, 4) as i:
            d_x[i] = 0
        with ft.VarDef("d_t", (4,), "float32", "cache", "cpu") as d_t:
            with ft.For("i1", 0, 4) as i:
                d_t[i] = 0
            with ft.VarDef("t", (4,), "float32", "cache", "cpu") as t:
                with ft.For("i", 0, 4) as i:
                    t[i] = 0
                    with ft.For("j", 0, 4) as j:
                        t[i] += x[j] * w0[i, j]
                with ft.For("i", 3, -1, -1) as i:
                    with ft.For("j", 3, -1, -1) as j:
                        d_t[j] += d_y[i] * w1[i, j]
                        d_w1[i, j] = d_y[i] * t[j]
                    d_y[i] = 0
            with ft.For("i", 3, -1, -1) as i:
                with ft.For("j", 3, -1, -1) as j:
                    d_x[j] += d_t[i] * w0[i, j]
                    d_w0[i, j] = d_t[i] * x[j]
                d_t[i] = 0
    std = ft.pop_ast()

    assert std.match(ast)


def test_recomp_single_stmt():
    with ft.VarDef([
        ("x", (), "float32", "input", "cpu"),
        ("y", (), "float32", "output", "cpu"),
    ]) as (x, y):
        y[...] = ft.exp(x[...])
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("d_x", (), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x, d_x, d_y):
        d_x[()] = d_y[()] * ft.exp(x[()])
    std = ft.pop_ast()

    assert std.match(ast)


def test_multi_versions_in_recomp_1():

    @ft.transform(verbose=1)
    def func(x, y):
        x: ft.Var[(1024,), "float32", "input"]
        y: ft.Var[(1024,), "float32", "output"]
        for pn in range(256):
            z = ft.empty((), "float32")
            z[()] = 1.
            for fn in range(1024):
                z[()] = (z[()] + 1) * x[fn]
            y[pn] = 1. - z
        return y

    _, bwd, _, _ = ft.grad(func, ["x"], ["y"], set())
    print(bwd)
    bwd = ft.lower(bwd, verbose=1, skip_passes=['make_heap_alloc'])

    with ft.VarDef([("x", (1024,), "float32", "input"),
                    ("dx", (1024,), "float32", "output"),
                    ("dy", (1024,), "float32", "inout")]) as (x, dx, dy):
        with ft.For("i", 0, 1024) as i:
            dx[i] = 0
        with ft.For("pn", 255, -1, -1) as pn:
            with ft.VarDef("z.recomp", (1024,), "float32", "cache") as z_recomp:
                with ft.VarDef("z", (), "float32", "cache") as z:
                    z[()] = 1.
                    with ft.For("fn", 0, 1024) as fn:
                        z_recomp[fn] = z[()]
                        z[()] = (z_recomp[fn] + 1) * x[fn]
                with ft.VarDef("dz", (), "float32", "cache") as dz:
                    dz[()] = -1 * dy[pn]
                    with ft.For("fn", 1023, -1, -1) as fn:
                        with ft.VarDef("dz.old", (), "float32",
                                       "cache") as dz_old:
                            dz_old[()] = dz[()]
                            dz[()] = dz_old[()] * x[fn]
                            dx[fn] += dz_old[()] * (z_recomp[fn] + 1)

    std = ft.pop_ast()
    assert std.match(bwd.body)


def test_multi_versions_in_recomp_2():
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
    ast = ft.lower(ast, verbose=1, skip_passes=['make_heap_alloc'])

    with ft.VarDef([
        ("x", (4,), "float32", "input", "cpu"),
        ("dx", (4,), "float32", "output", "cpu"),
        ("w", (4,), "float32", "input", "cpu"),
        ("dw", (4,), "float32", "output", "cpu"),
        ("dy", (), "float32", "inout", "cpu"),
    ]) as (x, dx, w, dw, dy):
        with ft.VarDef("ds", (), "float32", "cache", "cpu") as ds:
            ds[...] = 0
            with ft.VarDef("s_recomp", (4,), "float32", "cache",
                           "cpu") as s_recomp:
                with ft.VarDef("s", (), "float32", "cache", "cpu") as s:
                    s[...] = 0
                    with ft.For("i", 0, 4) as i:
                        s[...] += w[i]
                        s_recomp[i] = s[...]
                with ft.For("i", 3, -1, -1) as i:
                    ds[...] += dy[...] * x[i]
                    dx[i] = dy[...] * s_recomp[i]
                    dw[i] = ds[...]
    std = ft.pop_ast()
    assert std.match(ast)


def test_tape_1():
    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, x3, y):
        ft.MarkLabel("V_t")
        with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
            t[()] = x1[()] + x2[()]
            y[()] = t[()] * x3[()]
    ast = ft.pop_ast(verbose=True)
    forward, backward, _, _, _ = ft.grad_body(ast, ["x1", "x2", "x3"], ["y"],
                                              ["V_t"])
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ft.lower(forward)
    backward = ft.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ft.VarDef([
        ("d_x1", (), "float32", "output", "cpu"),
        ("d_x2", (), "float32", "output", "cpu"),
        ("x3", (), "float32", "input", "cpu"),
        ("d_x3", (), "float32", "output", "cpu"),
        ("d_y", (), "float32", "inout", "cpu"),
    ]) as (d_x1, d_x2, x3, d_x3, d_y):
        with ft.VarDef("t", (), "float32", "input", "cpu") as t:
            with ft.VarDef("d_t", (), "float32", "cache", "cpu") as d_t:
                d_t[()] = d_y[()] * x3[()]
                d_x3[()] = d_y[()] * t[()]
                d_x1[()] = d_t[()]
                d_x2[()] = d_t[()]
    std = ft.pop_ast()

    assert std.match(backward)


def test_tape_2():
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x1, x2, x3,
                                                                  y):
        with ft.For("i", 0, 4) as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
                t[()] = x1[i] + x2[i]
                y[i] = t[()] * x3[i]
    ast = ft.pop_ast(verbose=True)
    forward, backward, _, _, _ = ft.grad_body(ast, ["x1", "x2", "x3"], ["y"],
                                              ["V_t"])
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ft.lower(forward)
    backward = ft.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ft.VarDef([("d_x1", (4,), "float32", "output", "cpu"),
                    ("d_x2", (4,), "float32", "output", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("d_x3", (4,), "float32", "output", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu")
                   ]) as (d_x1, d_x2, x3, d_x3, d_y):
        with ft.VarDef("t.tape", (4,), "float32", "input", "cpu") as t:
            with ft.For("i", 3, -1, -1) as i:
                with ft.VarDef("d_t", (), "float32", "cache", "cpu") as d_t:
                    d_t[()] = d_y[i] * x3[i]
                    d_x3[i] = d_y[i] * t[i]
                    d_x1[i] = d_t[()]
                    d_x2[i] = d_t[()]
    std = ft.pop_ast()

    assert std.match(backward)


def test_tape_3():
    with ft.VarDef([("x1", (4, 5, 6), "float32", "input", "cpu"),
                    ("x2", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6), "float32", "output", "cpu")]) as (x1, x2, y):
        with ft.For("i", 0, 4, label="Li") as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (6,), "float32", "cache", "cpu") as t:
                with ft.For("k", 0, 6, label="Lk0") as k:
                    t[k] = 0
                with ft.For("j", 0, 5, label="Lj") as j:
                    with ft.For("k", 0, 6, label="Lk1") as k:
                        t[k] += x1[i, j, k]
                with ft.For("k", 0, 6, label="Lk2"):
                    y[i, k] = 0
                    with ft.For("j", 0, 5, label="Lj") as j:
                        y[i, k] += t[k] * x2[i, j, k]
    ast = ft.pop_ast(verbose=True)
    forward, backward, _, _, _ = ft.grad_body(ast, ["x2"], ["y"], ["V_t"])
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ft.lower(forward)
    backward = ft.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ft.VarDef([("d_x2", (4, 5, 6), "float32", "output", "cpu"),
                    ("d_y", (4, 6), "float32", "inout", "cpu")]) as (d_x2, d_y):
        with ft.VarDef("t", (4, 6), "float32", "input", "cpu") as t:
            with ft.For("i", 3, -1, -1, label="Li") as i:
                with ft.For("k", 5, -1, -1, label="Lk2"):
                    with ft.For("j", 4, -1, -1, label="Lj") as j:
                        d_x2[i, j, k] = d_y[i, k] * t[i, k]
                    d_y[i, k] = 0
    std = ft.pop_ast()

    assert std.match(backward)


def test_tape_4():
    with ft.VarDef([("x", (100,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        ft.MarkLabel("V_t")
        with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
            t[()] = 1
            with ft.For("i", 0, 100) as i:
                t[()] = t[()] * x[i] + 1
            y[()] = t[()]
    ast = ft.pop_ast(verbose=True)
    forward, backward, _, _, _ = ft.grad_body(ast, ["x"], ["y"], ["V_t"])
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ft.lower(forward)
    backward = ft.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ft.VarDef([("x", (100,), "float32", "input", "cpu"),
                    ("d_x", (100,), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x, d_x, d_y):
        with ft.VarDef([("t", (101,), "float32", "input", "cpu"),
                        ("d_t", (), "float32", "cache", "cpu")]) as (t, d_t):
            d_t[()] = d_y[()]
            with ft.For("i", 99, -1, -1) as i:
                with ft.VarDef("d_t_old", (), "float32", "cache",
                               "cpu") as d_t_old:
                    d_t_old[()] = d_t[()]
                    d_t[()] = d_t_old[()] * x[i]
                    d_x[i] = d_t_old[()] * t[i]
    std = ft.pop_ast()

    assert std.match(backward)


def test_tape_5():
    with ft.VarDef([("x", (100, 4), "float32", "input", "cpu"),
                    ("y", (256,), "float32", "output", "cpu"),
                    ("u", (256, 256), "float32", "input", "cpu")]) as (x, y, u):
        ft.MarkLabel("h")
        with ft.VarDef("h", (256,), "float32", "cache", "cpu") as h:
            ft.MarkLabel("f")
            with ft.VarDef("f", (256,), "float32", "cache", "cpu") as f:
                with ft.For("l", 0, 256, label="Ll0") as l:
                    h[l] = 0
                with ft.For("k", 0, 100, label="Lk") as k:
                    with ft.For("l", 0, 256, label="Ll1") as l:
                        f[l] = 0
                        with ft.For("j", 0, 256, label="Lj") as j:
                            f[l] += u[j, l] * h[j]
                    with ft.For("l", 0, 256, label="Ll2") as l:
                        h[l] = f[l]
                with ft.For("i", 0, 256) as i:
                    y[i] = h[i]

    ast = ft.pop_ast(verbose=True)
    forward, backward, _, _, _ = ft.grad_body(ast, ["x", "u"], ["y"],
                                              ["h", "f"])
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ft.lower(forward)
    backward = ft.lower(backward, skip_passes=['make_heap_alloc'])
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ft.VarDef([("x.grad", (100, 4), "float32", "output", "cpu"),
                    ("y.grad", (256,), "float32", "inout", "cpu"),
                    ("u", (256, 256), "float32", "input", "cpu"),
                    ("u.grad", (256, 256), "float32", "output", "cpu"),
                    ("h.tape", (101, 256), "float32", "input", "cpu")
                   ]) as (x_grad, dy, u, du, h_tape):
        with ft.For(".x.grad.i0", 0, 100) as _x_grad_i0:
            with ft.For(".x.grad.i1", 0, 4) as _x_grad_i1:
                x_grad[_x_grad_i0, _x_grad_i1] = 0
        with ft.For(".u.grad.i0", 0, 256) as _du_i0:
            with ft.For(".u.grad.i1", 0, 256) as _du_i1:
                du[_du_i0, _du_i1] = 0
        with ft.VarDef("f.grad", (256,), "float32", "cache", "cpu") as df:
            with ft.For(".f.grad.i0", 0, 256) as _df_i0:
                df[_df_i0] = 0
            with ft.VarDef("h.grad", (256,), "float32", "cache", "cpu") as dh:
                with ft.For("i", 255, -1, -1) as i:
                    dh[i] = dy[i]
                with ft.For("k", 99, -1, -1) as k:
                    with ft.For("l", 255, -1, -1) as l:
                        df[l] += dh[l]
                        dh[l] = 0
                    with ft.For("l", 255, -1, -1) as l:
                        with ft.For("j", 255, -1, -1) as j:
                            du[j, l] += df[l] * h_tape[k, j]
                            dh[j] += df[l] * u[j, l]
                        df[l] = 0
    std = ft.pop_ast()

    assert std.match(backward)


def test_tape_6():
    with ft.VarDef([("x1", (4, 2, 4), "float32", "input", "cpu"),
                    ("x2", (4, 2, 4), "float32", "input", "cpu"),
                    ("x3", (4, 2, 4), "float32", "input", "cpu"),
                    ("y", (4, 2, 4), "float32", "output", "cpu")]) as (x1, x2,
                                                                       x3, y):
        with ft.For("i", 0, 4) as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (2, 4), "float32", "cache", "cpu") as t:
                # When taping `t`, no need to distinguish the following two loops
                # by versions
                with ft.For("j", 0, 4) as j:
                    t[0, j] = x1[i, 0, j] + x2[i, 0, j]
                    y[i, 0, j] = t[0, j] * x3[i, 0, j]
                with ft.For("j", 0, 4) as j:
                    t[1, j] = x1[i, 1, j] + x2[i, 1, j]
                    y[i, 1, j] = t[1, j] * x3[i, 1, j]

    ast = ft.pop_ast(verbose=True)
    forward, backward, _, _, _ = ft.grad_body(ast, ["x1", "x2", "x3"], ["y"],
                                              ["V_t"])
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ft.lower(forward)
    backward = ft.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ft.VarDef([
        ("d_x1", (4, 2, 4), "float32", "output", "cpu"),
        ("d_x2", (4, 2, 4), "float32", "output", "cpu"),
        ("x3", (4, 2, 4), "float32", "input", "cpu"),
        ("d_x3", (4, 2, 4), "float32", "output", "cpu"),
        ("d_y", (4, 2, 4), "float32", "inout", "cpu"),
        ("t_tape", (4, 2, 4), "float32", "input", "cpu"),
    ]) as (d_x1, d_x2, x3, d_x3, d_y, t_tape):
        with ft.For("i", 3, -1, -1) as i:
            with ft.VarDef("d_t", (2, 4), "float32", "cache", "cpu") as d_t:
                with ft.For("j", 3, -1, -1) as j:
                    d_t[1, j] = d_y[i, 1, j] * x3[i, 1, j]
                    d_x3[i, 1, j] = d_y[i, 1, j] * t_tape[i, 1, j]
                    d_x1[i, 1, j] = d_t[1, j]
                    d_x2[i, 1, j] = d_t[1, j]
                with ft.For("j", 3, -1, -1) as j:
                    d_t[0, j] = d_y[i, 0, j] * x3[i, 0, j]
                    d_x3[i, 0, j] = d_y[i, 0, j] * t_tape[i, 0, j]
                    d_x1[i, 0, j] = d_t[0, j]
                    d_x2[i, 0, j] = d_t[0, j]
    std = ft.pop_ast()

    assert std.match(backward)


def test_hoist_tape_out_of_loop():
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x1, x2, x3,
                                                                  y):
        with ft.For("i", 0, 4) as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
                t[()] = x1[i] + x2[i]
                y[i] = t[()] * x3[i]
    func = ft.Func("main", ["x1", "x2", "x3", "y"], [], ft.pop_ast())
    print(func)
    forward, backward, _, _ = ft.grad_(func, ["x1", "x2", "x3"], ["y"], ["V_t"])
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ft.lower(forward)
    backward = ft.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x1, x2, x3,
                                                                  y):
        with ft.VarDef("t_tape", (4,), "float32", "output", "cpu") as t_tape:
            # Here out of the loop
            with ft.For("i", 0, 4) as i:
                with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
                    t[()] = x1[i] + x2[i]
                    t_tape[i] = t[()]
                    y[i] = t[()] * x3[i]
    std = ft.pop_ast()

    assert std.match(forward.body)


def test_use_tape_in_cond():
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x1, x2, x3,
                                                                  y):
        with ft.For("i", 0, 4) as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
                t[()] = x1[i] + x2[i]
                with ft.If(t[()] >= 0):
                    y[i] = t[()] * x3[i]
                with ft.Else():
                    y[i] = t[()]
    func = ft.Func("main", ["x1", "x2", "x3", "y"], [], ft.pop_ast())
    print(func)
    forward, backward, _, _ = ft.grad_(func, ["x1", "x2", "x3"], ["y"], ["V_t"])
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ft.lower(forward)
    backward = ft.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ft.VarDef([("d_x1", (4,), "float32", "output", "cpu"),
                    ("d_x2", (4,), "float32", "output", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("d_x3", (4,), "float32", "output", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu")
                   ]) as (d_x1, d_x2, x3, d_x3, d_y):
        with ft.VarDef("t.tape", (4,), "float32", "input", "cpu") as t:
            with ft.For("i0", 0, 4) as i:
                d_x3[i] = 0
            with ft.For("i", 3, -1, -1) as i:
                with ft.VarDef("d_t", (), "float32", "cache", "cpu") as d_t:
                    with ft.If(t[i] >= 0):
                        d_t[()] = d_y[i] * x3[i]
                        d_x3[i] = d_y[i] * t[i]
                    with ft.Else():
                        d_t[()] = d_y[i]
                    d_x1[i] = d_t[()]
                    d_x2[i] = d_t[()]
    std = ft.pop_ast()

    assert std.match(backward.body)


def test_use_tape_in_index():

    @ft.transform(verbose=1)
    def test(w, x, y):
        w: ft.Var[(3,), "float32"]
        x: ft.Var[(25,), "float32"]
        y: ft.Var[(3,), "float32", "output"]
        for i in range(3):
            #! label: V
            w_x = ft.empty((), "int32")
            w_x[()] = w[i] * 5
            y[i] = x[w_x]
        return y

    forward, backward, _, _ = ft.grad_(test,
                                       set(["x"]),
                                       set(["y"]),
                                       tapes=["V"],
                                       verbose=2)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ft.lower(forward)
    backward = ft.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    @ft.transform
    def expected(w, x, w_x_tape, dx, dy):
        dx: ft.Var[(25,), "float32", "output"]
        dy: ft.Var[(3,), "float32", "inout"]
        w_x_tape: ft.Var[(3,), "int32", "input"]
        for k in range(25):
            dx[k] = 0
        for i in range(2, -1, -1):
            dx[w_x_tape[i]] += dy[i]

    assert expected.body.match(backward.body)


def test_use_a_taped_var_to_recompute_another_var():
    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("w", (), "float32", "output", "cpu")]) as (x, w):
        with ft.VarDef("z", (), "float32", "cache", "cpu") as z:
            z[...] = 0
            with ft.If(x[...] > 0):
                ft.MarkLabel("V_y")
                with ft.VarDef("y", (), "float32", "cache", "cpu") as y:
                    y[...] = x[...] * x[...]
                    z[...] = y[...] * y[...]
            w[...] = z[...] * z[...]
    func = ft.Func("main", ["x", "y"], [], ft.pop_ast())
    print(func)
    forward, backward, _, _ = ft.grad_(func, ["x"], ["w"], ["V_y"])
    print("Backward:")
    print(backward)
    backward = ft.lower(backward)
    print("Backward:")
    print(backward)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("x.grad", (), "float32", "output", "cpu"),
                    ("w.grad", (), "float32", "inout", "cpu")]) as (x, dx, dw):
        with ft.VarDef("y", (), "float32", "input", "cpu") as y:
            dx[...] = 0
            with ft.VarDef("z", (), "float32", "cache", "cpu") as z:
                z[...] = 0
                with ft.If(x[...] > 0):
                    z[...] = ft.square(y[...])
                    dx[...] = 8 * y[...] * dw[...] * z[...] * x[...]
    std = ft.pop_ast()

    assert std.match(backward.body)


def test_recompute_using_another_recomputed_var():

    @ft.transform(verbose=1)
    def func(a: ft.Var[(10,), "float32"]):
        b = ft.empty((10,), "float32")
        for i in range(10):
            b[i] = a[i] * a[i]
        c = ft.empty((10,), "float32")
        for i in range(10):
            t = ft.empty((), "float32")
            # t MUST BE RECOMPUTED USING THE OLD b BEFORE b IS UPDATED
            t[...] = b[i] * b[i]
            c[i] = t[...] * t[...]
            b[i] += 1  # HERE
        d = ft.empty((10,), "float32")
        for i in range(10):
            d[i] = c[i] * c[i] * b[i]
        return d

    _, bwd, _, _ = ft.grad(func, ["a"], ["d"], set())
    print(bwd)
    bwd = ft.lower(bwd, verbose=1)

    with ft.VarDef([("a", (10,), "float32", "input", "cpu"),
                    ("a.grad", (10,), "float32", "output", "cpu"),
                    ("d.grad", (10,), "float32", "inout", "cpu")]) as (a, da,
                                                                       dd):
        with ft.VarDef("b", (10,), "float32", "cache", "cpu") as b:
            with ft.For("i", 0, 10) as i:
                b[i] = ft.square(a[i])
            with ft.VarDef("b.recomp", (1, 10), "float32", "cache",
                           "cpu") as b_recomp:
                with ft.VarDef("c", (10,), "float32", "cache", "cpu") as c:
                    with ft.For("i_1", 0, 10) as i:
                        b_recomp[0, i] = b[i]
                        c[i] = ft.square(ft.square(b_recomp[0, i]))
                        b[i] += 1
                    with ft.For("i", 9, -1, -1) as i:
                        gradient_of_bi_in_di = dd[i] * ft.square(c[i])
                        # USE NEW b HERE
                        # gradient_of_ci_in_di = 2 * dd[i] * b[i] * c[i]
                        # USE OLD b_recomp HERE
                        # dt = (4 * ft.square(b_recomp[0, i]) *
                        #       gradient_of_ci_in_di * b_recomp[0, i])
                        dt = (8 * ft.square(b_recomp[0, i]) * dd[i] * b[i] *
                              c[i] * b_recomp[0, i])
                        da[i] = 2 * (gradient_of_bi_in_di + dt) * a[i]
    std = ft.pop_ast()

    assert std.match(bwd.body)


def test_tape_vars_with_the_same_name():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            ft.MarkLabel("Vt1")
            with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
                t[...] = ft.exp(x[i])
                y[i] = t[...] * t[...]
        with ft.For("i", 0, 4) as i:
            ft.MarkLabel("Vt2")
            with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
                t[...] = ft.exp(y[i])
                y[i] = t[...] * t[...]
    func = ft.Func("main", ["x", "y"], [], ft.pop_ast())
    print(func)
    forward, backward, _, _ = ft.grad_(func, ["x"], ["y"], ["Vt1", "Vt2"])
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ft.lower(forward)
    backward = ft.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    tape_output = list(
        filter(lambda v: v.endswith(".tape"),
               map(lambda ret: ret.name, forward.returns)))
    tape_input = list(
        filter(lambda v: v.endswith(".tape"),
               map(lambda param: param.name, backward.params)))
    # There should be 2 tapes in different names, and the output and input
    # tape name should be consistent
    assert len(tape_output) == 2
    assert tape_output[0] != tape_output[1]
    assert sorted(tape_output) == sorted(tape_input)


def test_single_version_tape():
    # Although b is only read once, but it is then overwritten, we still
    # need a tape

    @ft.transform(verbose=1)
    def func(a: ft.Var[(10,), "float32"]):
        #! label: V_b
        b = ft.empty((10,), "float32")
        for i in range(10):
            b[i] = a[i] * a[i]
        c = ft.empty((10,), "float32")
        for i in range(10):
            c[i] = b[i] * b[i]
            b[i] += 1  # HERE!!
        return b, c

    _, bwd, _, _ = ft.grad(func, ["a"], ["b", "c"], set(["V_b"]))
    print(bwd)
    bwd = ft.lower(bwd, verbose=1)

    @ft.transform
    def expected(a, d_a, b_tape, d_c):
        a: ft.Var[(10,), "float32", "input"]
        d_a: ft.Var[(10,), "float32", "output"]
        b_tape: ft.Var[(1, 10), "float32", "input"]
        d_b: ft.Var[(10,), "float32", "inout"]
        d_c: ft.Var[(10,), "float32", "inout"]
        for i in range(9, -1, -1):
            # HERE WE STILL NEED A TAPE
            d_b[i] += 2 * d_c[i] * b_tape[0, i]
        for i in range(9, -1, -1):
            d_a[i] = 2 * a[i] * d_b[i]
            d_b[i] = 0

    assert expected.body.match(bwd.body)


def test_tape_mode_all():
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x1, x2, x3,
                                                                  y):
        with ft.VarDef("t", (4,), "float32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = x1[i] + x2[i]
            with ft.For("i", 0, 4) as i:
                with ft.VarDef("u", (), "float32", "cache", "cpu") as u:
                    u[()] = x2[i] + x3[i]
                    y[i] = u[()] * t[i]
    ast = ft.pop_ast(verbose=True)
    forward, backward, _, _, _ = ft.grad_body(ast, ["x1", "x2", "x3"], ["y"],
                                              ft.GradTapeMode.All)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ft.lower(forward)
    backward = ft.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ft.VarDef([
        ("d_x1", (4,), "float32", "output", "cpu"),
        ("d_x2", (4,), "float32", "output", "cpu"),
        ("d_x3", (4,), "float32", "output", "cpu"),
        ("d_y", (4,), "float32", "inout", "cpu"),
    ]) as (d_x1, d_x2, d_x3, d_y):
        with ft.VarDef([("t.tape", (4,), "float32", "input", "cpu"),
                        ("u.tape", (4,), "float32", "input", "cpu"),
                        ("d_t", (4,), "float32", "cache", "cpu")]) as (t, u,
                                                                       d_t):
            with ft.For("i", 3, -1, -1) as i:
                with ft.VarDef("d_u", (), "float32", "cache", "cpu") as d_u:
                    d_u[()] = d_y[i] * t[i]
                    d_t[i] = d_y[i] * u[i]
                    d_x2[i] = d_u[()]
                    d_x3[i] = d_u[()]
            with ft.For("i", 3, -1, -1) as i:
                d_x1[i] = d_t[i]
                d_x2[i] += d_t[i]
    std = ft.pop_ast()

    assert std.match(backward)


def test_tape_mode_nothing():
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x1, x2, x3,
                                                                  y):
        with ft.VarDef("t", (4,), "float32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = x1[i] + x2[i]
            with ft.For("i", 0, 4) as i:
                with ft.VarDef("u", (), "float32", "cache", "cpu") as u:
                    u[()] = x2[i] + x3[i]
                    y[i] = u[()] * t[i]
    ast = ft.pop_ast(verbose=True)
    forward, backward, _, _, _ = ft.grad_body(ast, ["x1", "x2", "x3"], ["y"],
                                              ft.GradTapeMode.Nothing)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ft.lower(forward)
    backward = ft.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("d_x1", (4,), "float32", "output", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("d_x2", (4,), "float32", "output", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("d_x3", (4,), "float32", "output", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, d_y):
        with ft.VarDef("d_t", (4,), "float32", "cache", "cpu") as d_t:
            with ft.For("i", 3, -1, -1) as i:
                with ft.VarDef("d_u", (), "float32", "cache", "cpu") as d_u:
                    d_u[()] = d_y[i] * (x1[i] + x2[i])
                    d_t[i] = d_y[i] * (x2[i] + x3[i])
                    d_x2[i] = d_u[()]
                    d_x3[i] = d_u[()]
            with ft.For("i", 3, -1, -1) as i:
                d_x1[i] = d_t[i]
                d_x2[i] += d_t[i]
    std = ft.pop_ast()

    assert std.match(backward)


def test_tape_mode_no_reuse_only():
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x1, x2, x3,
                                                                  y):
        with ft.VarDef("t", (4,), "float32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = x1[i] + x2[i]
            with ft.For("i", 0, 4) as i:
                with ft.VarDef("u", (), "float32", "cache", "cpu") as u:
                    u[()] = x2[i] + x3[i]
                    y[i] = u[()] * t[i]
    ast = ft.pop_ast(verbose=True)
    forward, backward, _, _, _ = ft.grad_body(ast, ["x1", "x2", "x3"], ["y"],
                                              ft.GradTapeMode.NoReuseOnly)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ft.lower(forward)
    backward = ft.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ft.VarDef([("d_x1", (4,), "float32", "output", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("d_x2", (4,), "float32", "output", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("d_x3", (4,), "float32", "output", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu")
                   ]) as (d_x1, x2, d_x2, x3, d_x3, d_y):
        with ft.VarDef([("t.tape", (4,), "float32", "input", "cpu"),
                        ("d_t", (4,), "float32", "cache", "cpu")]) as (t, d_t):
            with ft.For("i", 3, -1, -1) as i:
                with ft.VarDef("d_u", (), "float32", "cache", "cpu") as d_u:
                    d_u[()] = d_y[i] * t[i]
                    d_t[i] = d_y[i] * (x2[i] + x3[i])
                    d_x2[i] = d_u[()]
                    d_x3[i] = d_u[()]
            with ft.For("i", 3, -1, -1) as i:
                d_x1[i] = d_t[i]
                d_x2[i] += d_t[i]
    std = ft.pop_ast()

    assert std.match(backward)


def test_no_unused_trival_tape():

    @ft.transform
    def test(x: ft.Var[(), "float32", "input"]):
        t = ft.empty((), "float32")
        t[...] = x[...] * 2
        u = ft.empty((), "float32")
        u[...] = x[...] * 3
        y = ft.empty((), "float32")
        # t and u's forward values are actually not needed
        y[...] = t[...] + u[...]
        return y

    fwd, bwd, input_grads, output_grads = ft.grad(test, ["x"], [ft.Return()],
                                                  ft.GradTapeMode.All,
                                                  verbose=1)

    assert len(fwd.returns) == 1  # `y`. No `u or `v`
    assert len(bwd.params) == 2  # `x` and `y.grad`. No `u` or `v`


def test_no_unused_non_trivial_tape():

    @ft.transform
    def test(x: ft.Var[(4,), "float32", "input"]):
        y = ft.empty((4,), "float32")
        for i in range(4):
            t = ft.empty((), "float32")
            t[...] = x[i] * 2
            u = ft.empty((), "float32")
            u[...] = x[i] * 3
            # t and u's forward values are actually not needed
            y[i] = t[...] + u[...]
        return y

    fwd, bwd, input_grads, output_grads = ft.grad(test, ["x"], [ft.Return()],
                                                  ft.GradTapeMode.All,
                                                  verbose=1)

    assert len(fwd.returns) == 1  # `y`. No `u or `v`
    assert len(bwd.params) == 2  # `x` and `y.grad`. No `u` or `v`


def test_no_deps():

    @ft.transform
    def test(ptr, edge1, edge2):
        ptr: ft.Var[(11,), "int32", "input", "cpu"]
        edge1: ft.Var[(50,), "float32", "input", "cpu"]
        edge2: ft.Var[(50,), "float32", "output", "cpu"]
        #! label: Li
        #! no_deps: edge2
        for i in range(10):
            for j in range(ptr[i], ptr[i + 1]):
                edge2[j] = edge1[j] + i

    print(test)
    _, backward, _, _ = ft.grad_(test, ["edge1"], ["edge2"], set())
    print(backward)
    s = ft.Schedule(backward)
    s.parallelize("$grad{Li}", "openmp")  # No exception here
    print(s.ast())


def test_error_input_not_found():

    @ft.transform
    def test(x, y):
        x: ft.Var[(), "float32", "input", "cpu"]
        y: ft.Var[(), "float32", "output", "cpu"]
        y[...] = x[...] * 2

    with pytest.raises(ft.InvalidAutoGrad):
        ft.grad_(test, ["error"], ["y"], set())


def test_error_output_not_found():

    @ft.transform
    def test(x, y):
        x: ft.Var[(), "float32", "input", "cpu"]
        y: ft.Var[(), "float32", "output", "cpu"]
        y[...] = x[...] * 2

    with pytest.raises(ft.InvalidAutoGrad):
        ft.grad_(test, ["x"], ["error"], set())
