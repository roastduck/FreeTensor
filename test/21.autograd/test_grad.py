import pytest
import freetensor as ft
import numpy as np


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


def test_cast():
    with ft.VarDef([
        ("x1", (), "float32", "input", "cpu"),
        ("x2", (), "float32", "input", "cpu"),
        ("x3", (), "float32", "input", "cpu"),
        ("y", (), "float64", "output", "cpu"),
    ]) as (x1, x2, x3, y):
        y[()] = ft.cast(x1[()] + x2[()], "float64") * x3[()]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x1", "x2", "x3"], ["y"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("d_x1", (), "float32", "output", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("d_x2", (), "float32", "output", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("d_x3", (), "float32", "output", "cpu"),
                    ("d_y", (), "float64", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, d_y):
        d_x1[()] = ft.cast(d_y[()] * x3[()], "float32")
        d_x2[()] = ft.cast(d_y[()] * x3[()], "float32")
        d_x3[()] = ft.cast(d_y[()] * ft.cast(x1[()] + x2[()], "float64"),
                           "float32")
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
        ("y1", (4,), "float32>0", "input", "cpu"),
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
        with ft.VarDef("t.tape", (4,), "float32>0", "input", "cpu") as t:
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
                    ("y_tape", (2,), "float32>0", "input", "cpu"),
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
    _, ast, _, _, _ = ft.grad_body(ast, ["x", "w"], ["y"], set(), invert=False)
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

    _, bwd, _, _ = ft.grad(func, ["a"], ["d"], set(), invert=False)
    print(bwd)
    bwd = ft.lower(bwd, verbose=1)

    with ft.VarDef([("a", (10,), "float32", "input", "cpu"),
                    ("a.grad", (10,), "float32", "output", "cpu"),
                    ("d.grad", (10,), "float32", "inout", "cpu")]) as (a, da,
                                                                       dd):
        with ft.VarDef("b", (10,), "float32>=0", "cache", "cpu") as b:
            with ft.For("i", 0, 10) as i:
                b[i] = ft.square(a[i])
            with ft.VarDef("b.recomp", (1, 10), "float32>=0", "cache",
                           "cpu") as b_recomp:
                with ft.VarDef("c", (10,), "float32>=0", "cache", "cpu") as c:
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


def test_inlined_invoke_func_from_ad():

    def test(a: ft.Var[(4,), "float32"], b: ft.Var[(4,), "float32"]):
        y = ft.zeros((), "float32")
        for i in range(4):
            y[()] += a[i] * b[i]
        return y

    fwd, bwd, input_grads, output_grads = ft.grad(test, ['b'], [ft.Return()],
                                                  ft.GradTapeMode.Nothing,
                                                  tape_in_closure=False,
                                                  verbose=1)

    @ft.transform(verbose=1)
    def custom_bwd(a: ft.Var[(4,), "float32"], b: ft.Var[(4,), "float32"],
                   dzdy: ft.Var[(), "float32"]):
        b_grad = bwd(a, b, **{output_grads[ft.Return()]: dzdy})
        b_grad[...] += 1
        return b_grad

    fwd = ft.optimize(fwd)
    custom_bwd = ft.optimize(custom_bwd)

    a = ft.array([0, 1, 2, 3], dtype="float32")
    b = ft.array([3, 2, 1, 0], dtype="float32")
    y = fwd(a, b)
    dzdy = ft.array(1, dtype='float32')
    dzdb = custom_bwd(a, b, dzdy)

    assert y.numpy() == 4
    assert np.array_equal(dzdb.numpy(), [1, 2, 3, 4])


def test_specify_param_and_return_by_position():

    @ft.transform
    def test(a, b):
        a: ft.Var[(4,), "float32", "input", "cpu"]
        b: ft.Var[(4,), "float32", "input", "cpu"]
        x = a + b
        y = a * b
        return x, y

    fwd, bwd, _, _ = ft.grad(test, [ft.Parameter(0)], [ft.Return(1)])
    fwd = ft.optimize(fwd)
    bwd = ft.optimize(bwd)
    a = ft.array([1, 2, 3, 4], dtype="float32")
    b = ft.array([5, 6, 7, 8], dtype="float32")
    one = ft.array([1, 1, 1, 1], dtype="float32")
    c, d = fwd(a, b)
    da = bwd(one)

    assert np.array_equal(da.numpy(), [5, 6, 7, 8])


def test_attach_backward():

    @ft.optimize
    @ft.grad(requires=['a', 'b'], provides=[ft.Return(1)], attach_backward=True)
    def test(a, b):
        a: ft.Var[(4,), "float32", "input", "cpu"]
        b: ft.Var[(4,), "float32", "input", "cpu"]
        x = a + b
        y = a * b
        return x, y

    a = ft.array([1, 2, 3, 4], dtype="float32")
    b = ft.array([5, 6, 7, 8], dtype="float32")
    one = ft.array([1, 1, 1, 1], dtype="float32")
    c, d = test(a, b)
    da, db = test.backward(
        **{test.output_name_to_gradient_name[ft.Return(1)]: one
          })[test.input_name_to_gradient_name['a'],
             test.input_name_to_gradient_name['b']]

    assert np.array_equal(da.numpy(), [5, 6, 7, 8])
    assert np.array_equal(db.numpy(), [1, 2, 3, 4])


def test_must_attach_backward_when_used_as_decorator():

    with pytest.raises(TypeError):

        @ft.grad(requires=['a', 'b'],
                 provides=[ft.Return(1)],
                 attach_backward=False)
        def f(a, b):
            a: ft.Var[(4, 5), "float32", "input", "cpu"]
            b: ft.Var[(5, 6), "float32", "input", "cpu"]
            return ft.libop.matmul(a, b)


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
