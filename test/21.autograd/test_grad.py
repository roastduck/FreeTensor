import freetensor as ft


def test_basic():
    with ft.VarDef([
        ("x1", (), "float32", "input", "cpu"),
        ("x2", (), "float32", "input", "cpu"),
        ("x3", (), "float32", "input", "cpu"),
        ("y", (), "float32", "output", "cpu"),
    ]) as (x1, x2, x3, y):
        y[()] = (x1[()] + x2[()]) * x3[()]
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
        y[()] = (x1[()] + x2[()]) * x3[()]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x1"], ["y"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("d_x1", (), "float32", "output", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x1, d_x1, x2,
                                                                 x3, d_y):
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
    std = ft.make_reduction(ft.pop_ast())

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
            ()] / (2 * ft.sqrt(x[()]))
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_use_forward_value_when_taped():
    with ft.VarDef("x", (4,), "float32", "input", "cpu") as x:
        ft.MarkNid("V_y1")
        with ft.VarDef("y1", (4,), "float32", "output", "cpu") as y1:
            ft.MarkNid("V_y2")
            with ft.VarDef("y2", (4,), "float32", "output", "cpu") as y2:
                with ft.For("i", 0, 4) as i:
                    ft.MarkNid("V_t")
                    with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
                        t[()] = 2 * x[i]
                        y1[i] = ft.exp(t[()])
                        y2[i] = ft.square(t[()])
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y1", "y2"],
                                   ["V_t", "V_y1", "V_y2"])
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("x", (4,), "float32", "input", "cpu"),
        ("d_x", (4,), "float32", "output", "cpu"),
        ("y1", (4,), "float32", "input", "cpu"),
        ("d_y1", (4,), "float32", "inout", "cpu"),
        ("y2", (4,), "float32", "input", "cpu"),
        ("d_y2", (4,), "float32", "inout", "cpu"),
    ]) as (x, d_x, y1, d_y1, y2, d_y2):
        with ft.For("i", 3, -1, -1) as i:
            with ft.VarDef("t.tape", (4,), "float32", "input", "cpu") as t:
                d_x[i] = 2 * (d_y1[i] * y1[i] + 2 * (d_y2[i] * t[i]))
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_use_taped_forward_value():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y1", (4,), "float32", "output", "cpu"),
                    ("y2", (4,), "float32", "output", "cpu")]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            ft.MarkNid("V_t")
            with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
                t[()] = ft.exp(x[i])
                y1[i] = 2 * t[()]
                y2[i] = 3 * t[()]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y1", "y2"], ["V_t"])
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("x", (4,), "float32", "input", "cpu"),
        ("d_x", (4,), "float32", "output", "cpu"),
        ("d_y1", (4,), "float32", "inout", "cpu"),
        ("d_y2", (4,), "float32", "inout", "cpu"),
    ]) as (x, d_x, d_y1, d_y2):
        with ft.For("i", 3, -1, -1) as i:
            with ft.VarDef("t.tape", (4,), "float32", "input", "cpu") as t:
                d_x[i] = ((d_y2[i] * 3) + (d_y1[i] * 2)) * t[i]
    std = ft.make_reduction(ft.pop_ast())

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
    std = ft.make_reduction(ft.pop_ast())

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

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("d_x", (4,), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x, d_x, d_y):
        with ft.For("i", 3, -1, -1) as i:
            with ft.VarDef("d_y.old", (), "float32", "cache", "cpu") as d_y_old:
                d_y_old[()] = d_y[()]
                d_y[()] = -1 * d_y_old[()]
                d_x[i] = d_y_old[()]
    std = ft.make_reduction(ft.pop_ast())

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

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("d_x", (4,), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x, d_x, d_y):
        with ft.For("i", 3, -1, -1) as i:
            d_x[i] = d_y[()]
            d_y[()] = 0
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_reduce_sum_quick_path():
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[()] += x[i]
    ast = ft.pop_ast(verbose=True)
    _, ast, _, _, _ = ft.grad_body(ast, ["x"], ["y"], set())
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("d_x", (4,), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x, d_x, d_y):
        with ft.For("i", 3, -1, -1) as i:
            d_x[i] = d_y[()]
    std = ft.make_reduction(ft.pop_ast())

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

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("d_x", (4,), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x, d_x, d_y):
        with ft.For("i", 1, -3, -1) as i:
            with ft.VarDef("d_y.old", (), "float32", "cache", "cpu") as d_y_old:
                d_y_old[()] = d_y[()]
                d_y[()] = -1 * d_y_old[()]
                d_x[i + 2] = d_y_old[()]
    std = ft.make_reduction(ft.pop_ast())

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
        with ft.For("i0", 3, -1, -1) as i:
            d_x[i] = 0
        with ft.VarDef("d_t", (4,), "float32", "cache", "cpu") as d_t:
            with ft.For("i1", 3, -1, -1) as i:
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
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_tape_1():
    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, x3, y):
        ft.MarkNid("V_t")
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

    with ft.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("d_x1", (), "float32", "output", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("d_x2", (), "float32", "output", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("d_x3", (), "float32", "output", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, d_y):
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
            ft.MarkNid("V_t")
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

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("d_x1", (4,), "float32", "output", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("d_x2", (4,), "float32", "output", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("d_x3", (4,), "float32", "output", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, d_y):
        with ft.For("i", 3, -1, -1) as i:
            with ft.VarDef("t.tape", (4,), "float32", "input", "cpu") as t:
                with ft.VarDef("d_t", (), "float32", "cache", "cpu") as d_t:
                    d_t[()] = d_y[i] * x3[i]
                    d_x3[i] = d_y[i] * t[i]
                    d_x1[i] = d_t[()]
                    d_x2[i] = d_t[()]
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(backward)


def test_tape_3():
    with ft.VarDef([("x1", (4, 5, 6), "float32", "input", "cpu"),
                    ("x2", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6), "float32", "output", "cpu")]) as (x1, x2, y):
        with ft.For("i", 0, 4, nid="Li") as i:
            ft.MarkNid("V_t")
            with ft.VarDef("t", (6,), "float32", "cache", "cpu") as t:
                with ft.For("k", 0, 6, nid="Lk0") as k:
                    t[k] = 0
                with ft.For("j", 0, 5, nid="Lj") as j:
                    with ft.For("k", 0, 6, nid="Lk1") as k:
                        t[k] += x1[i, j, k]
                with ft.For("k", 0, 6, nid="Lk2"):
                    y[i, k] = 0
                    with ft.For("j", 0, 5, nid="Lj") as j:
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

    with ft.VarDef([("x1", (4, 5, 6), "float32", "input", "cpu"),
                    ("x2", (4, 5, 6), "float32", "input", "cpu"),
                    ("d_x2", (4, 5, 6), "float32", "output", "cpu"),
                    ("d_y", (4, 6), "float32", "inout", "cpu")]) as (x1, x2,
                                                                     d_x2, d_y):
        with ft.For("i", 3, -1, -1, nid="Li") as i:
            with ft.VarDef("t", (4, 6), "float32", "input", "cpu") as t:
                with ft.For("k", 5, -1, -1, nid="Lk2"):
                    with ft.For("j", 4, -1, -1, nid="Lj") as j:
                        d_x2[i, j, k] = d_y[i, k] * t[i, k]
                    d_y[i, k] = 0
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(backward)


def test_tape_4():
    with ft.VarDef([("x", (100,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        ft.MarkNid("V_t")
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
            d_t[()] = 0
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(backward)


def test_tape_5():
    with ft.VarDef([("x", (100, 4), "float32", "input", "cpu"),
                    ("y", (256,), "float32", "output", "cpu"),
                    ("u", (256, 256), "float32", "input", "cpu")]) as (x, y, u):
        ft.MarkNid(":h")
        with ft.VarDef("h", (256,), "float32", "cache", "cpu") as h:
            ft.MarkNid(":f")
            with ft.VarDef("f", (256,), "float32", "cache", "cpu") as f:
                with ft.For("l", 0, 256, nid="Ll0") as l:
                    h[l] = 0
                with ft.For("k", 0, 100, nid="Lk") as k:
                    with ft.For("l", 0, 256, nid="Ll1") as l:
                        f[l] = 0
                        with ft.For("j", 0, 256, nid="Lj") as j:
                            f[l] += u[j, l] * h[j]
                    with ft.For("l", 0, 256, nid="Ll2") as l:
                        h[l] = f[l]
                with ft.For("i", 0, 256) as i:
                    y[i] = h[i]

    ast = ft.pop_ast(verbose=True)
    forward, backward, _, _, _ = ft.grad_body(ast, ["x", "u"], ["y"],
                                              [":h", ":f"])
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

    with ft.VarDef([("x", (100, 4), "float32", "input", "cpu"),
                    ("x.grad", (100, 4), "float32", "output", "cpu"),
                    ("y.grad", (256,), "float32", "inout", "cpu"),
                    ("u", (256, 256), "float32", "input", "cpu"),
                    ("u.grad", (256, 256), "float32", "output", "cpu"),
                    ("h.tape", (101, 256), "float32", "input", "cpu"),
                    ("f.tape", (100, 256), "float32", "input", "cpu")
                   ]) as (x, x_grad, dy, u, du, h_tape, f_tape):
        with ft.For(".x.grad.i0", 99, -1, -1) as _x_grad_i0:
            with ft.For(".x.grad.i1", 3, -1, -1) as _x_grad_i1:
                x_grad[_x_grad_i0, _x_grad_i1] = 0
        with ft.For(".u.grad.i0", 255, -1, -1) as _du_i0:
            with ft.For(".u.grad.i1", 255, -1, -1) as _du_i1:
                du[_du_i0, _du_i1] = 0
        with ft.VarDef("h.grad", (256,), "float32", "cache", "cpu") as dh:
            with ft.VarDef("f.grad", (256,), "float32", "cache", "cpu") as df:
                with ft.For(".f.grad.i0", 255, -1, -1) as _df_i0:
                    df[_df_i0] = 0
                with ft.For("i", 255, -1, -1) as i:
                    dh[i] = dy[i]
                with ft.For("k", 99, -1, -1) as k:
                    with ft.For("l", 255, -1, -1) as l:
                        df[l] = df[l] + dh[l]
                        dh[l] = 0
                    with ft.For("l", 255, -1, -1) as l:
                        with ft.For("j", 255, -1, -1) as j:
                            du[j, l] = du[j, l] + df[l] * h_tape[k, j]
                            dh[j] = dh[j] + df[l] * u[j, l]
                        df[l] = 0
            with ft.For("l", 255, -1, -1) as l:
                dh[l] = 0
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(backward)


def test_hoist_tape_out_of_loop():
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x1, x2, x3,
                                                                  y):
        with ft.For("i", 0, 4) as i:
            ft.MarkNid("V_t")
            with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
                t[()] = x1[i] + x2[i]
                y[i] = t[()] * x3[i]
    func = ft.Func("main", ["x1", "x2", "x3", "y"], [], ft.pop_ast())
    print(func)
    forward, backward, _, _, _ = ft.grad_(func, ["x1", "x2", "x3"], ["y"],
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
            ft.MarkNid("V_t")
            with ft.VarDef("t", (), "float32", "cache", "cpu") as t:
                t[()] = x1[i] + x2[i]
                with ft.If(t[()] >= 0):
                    y[i] = t[()] * x3[i]
                with ft.Else():
                    y[i] = t[()]
    func = ft.Func("main", ["x1", "x2", "x3", "y"], [], ft.pop_ast())
    print(func)
    forward, backward, _, _, _ = ft.grad_(func, ["x1", "x2", "x3"], ["y"],
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

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("d_x1", (4,), "float32", "output", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("d_x2", (4,), "float32", "output", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("d_x3", (4,), "float32", "output", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, d_y):
        with ft.For("i0", 3, -1, -1) as i:
            d_x3[i] = 0
        with ft.For("i", 3, -1, -1) as i:
            with ft.VarDef([("t.tape", (4,), "float32", "input", "cpu"),
                            ("d_t", (), "float32", "cache", "cpu")]) as (t,
                                                                         d_t):
                d_t[()] = 0
                with ft.If(t[i] >= 0):
                    d_t[()] = d_y[i] * x3[i]
                    d_x3[i] = d_y[i] * t[i]
                    d_y[i] = 0
                    # FIXME: Why did we not remove this `= 0`?
                    # Bugs in analyze/deps that thinks two branches of the `If` depend on each other?
                with ft.Else():
                    d_t[()] = d_y[i]
                d_x1[i] = d_t[()]
                d_x2[i] = d_t[()]
    std = ft.pop_ast()

    assert std.match(backward.body)


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

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("d_x1", (4,), "float32", "output", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("d_x2", (4,), "float32", "output", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("d_x3", (4,), "float32", "output", "cpu"),
                    ("y", (4,), "float32", "input", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        with ft.VarDef([("t.tape", (4,), "float32", "input", "cpu"),
                        ("d_t", (4,), "float32", "cache", "cpu")]) as (t, d_t):
            with ft.For("i", 3, -1, -1) as i:
                with ft.VarDef("u.tape", (4,), "float32", "input", "cpu") as u:
                    with ft.VarDef("d_u", (), "float32", "cache", "cpu") as d_u:
                        d_u[()] = d_y[i] * t[i]
                        d_t[i] = d_y[i] * u[i]
                        d_x2[i] = d_u[()]
                        d_x3[i] = d_u[()]
            with ft.For("i", 3, -1, -1) as i:
                d_x1[i] = d_t[i]
                d_x2[i] += d_t[i]
    std = ft.make_reduction(ft.pop_ast())

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
    std = ft.make_reduction(ft.pop_ast())

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

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("d_x1", (4,), "float32", "output", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("d_x2", (4,), "float32", "output", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("d_x3", (4,), "float32", "output", "cpu"),
                    ("y", (4,), "float32", "input", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
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
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(backward)


def test_no_deps():

    @ft.transform
    def test(ptr, edge1, edge2):
        ptr: ft.Var[(11,), "int32", "input", "cpu"]
        edge1: ft.Var[(50,), "float32", "input", "cpu"]
        edge2: ft.Var[(50,), "float32", "output", "cpu"]
        #! nid: Li
        #! no_deps: edge2
        for i in range(10):
            for j in range(ptr[i], ptr[i + 1]):
                edge2[j] = edge1[j] + i

    print(test)
    _, backward, _, _, _ = ft.grad_(test, ["edge1"], ["edge2"], set())
    print(backward)
    s = ft.Schedule(backward)
    s.parallelize("Li", "openmp")  # No exception here
    print(s.ast())
