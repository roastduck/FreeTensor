import ir


def test_basic():
    with ir.VarDef([
        ("x1", (), "int32", "input", "cpu"),
        ("x2", (), "int32", "input", "cpu"),
        ("x3", (), "int32", "input", "cpu"),
        ("y", (), "int32", "output", "cpu"),
    ]) as (x1, x2, x3, y):
        y[()] = (x1[()] + x2[()]) * x3[()]
    ast = ir.pop_ast()
    print(ast)
    _, ast, _, _, _ = ir.grad(ast, set(["x1", "x2", "x3"]), set(["y"]), set())
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("d_x1", (), "int32", "output", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("d_x2", (), "int32", "output", "cpu"),
                    ("x3", (), "int32", "input", "cpu"),
                    ("d_x3", (), "int32", "output", "cpu"),
                    ("y", (), "int32", "input", "cpu"),
                    ("d_y", (), "int32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        with ir.VarDef("d_y.old", (), "int32", "cache", "cpu") as d_y_old:
            d_y_old[()] = d_y[()]
            d_y[()] = 0
            d_x1[()] = 0 + d_y_old[()] * x3[()]
            d_x2[()] = 0 + d_y_old[()] * x3[()]
            d_x3[()] = 0 + d_y_old[()] * (x1[()] + x2[()])
    std = ir.pop_ast()

    assert std.match(ast)


def test_branching_exprs():
    with ir.VarDef([
        ("x1", (), "int32", "input", "cpu"),
        ("x2", (), "int32", "input", "cpu"),
        ("y1", (), "int32", "output", "cpu"),
        ("y2", (), "int32", "output", "cpu"),
        ("y3", (), "int32", "output", "cpu"),
    ]) as (x1, x2, y1, y2, y3):
        y1[()] = ir.min(x1[()], x2[()])
        y2[()] = ir.max(x1[()], x2[()])
        y3[()] = ir.if_then_else(x1[()] > 0, x1[()], x2[()])
    ast = ir.pop_ast()
    print(ast)
    _, ast, _, _, _ = ir.grad(ast, set(["x1", "x2"]), set(["y1", "y2", "y3"]),
                              set())
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("d_x1", (), "int32", "output", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("d_x2", (), "int32", "output", "cpu"),
                    ("y1", (), "int32", "input", "cpu"),
                    ("d_y1", (), "int32", "inout", "cpu"),
                    ("y2", (), "int32", "input", "cpu"),
                    ("d_y2", (), "int32", "inout", "cpu"),
                    ("y3", (), "int32", "input", "cpu"),
                    ("d_y3", (), "int32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, y1, d_y1, y2, d_y2, y3, d_y3):
        with ir.VarDef("d_y3.old", (), "int32", "cache", "cpu") as d_y3_old:
            d_y3_old[()] = d_y3[()]
            d_y3[()] = 0
            with ir.VarDef("d_y2.old", (), "int32", "cache", "cpu") as d_y2_old:
                d_y2_old[()] = d_y2[()]
                d_y2[()] = 0
                with ir.VarDef("d_y1.old", (), "int32", "cache",
                               "cpu") as d_y1_old:
                    d_y1_old[()] = d_y1[()]
                    d_y1[()] = 0
                    d_x1[()] = ir.if_then_else(
                        x1[()] > 0, d_y3_old[()], 0) + ir.if_then_else(
                            x1[()] >= x2[()],
                            d_y2_old[()], 0) + ir.if_then_else(
                                x1[()] <= x2[()], d_y1_old[()], 0)
                    d_x2[()] = ir.if_then_else(
                        x1[()] <= 0, d_y3_old[()], 0) + ir.if_then_else(
                            x2[()] > x1[()], d_y2_old[()], 0) + ir.if_then_else(
                                x2[()] < x1[()], d_y1_old[()], 0)
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_math_funcs():
    with ir.VarDef([
        ("x", (), "int32", "input", "cpu"),
        ("y1", (), "int32", "output", "cpu"),
        ("y2", (), "int32", "output", "cpu"),
        ("y3", (), "int32", "output", "cpu"),
    ]) as (x, y1, y2, y3):
        y1[()] = ir.sqrt(x[()])
        y2[()] = ir.exp(x[()])
        y3[()] = ir.square(x[()])
    ast = ir.pop_ast()
    print(ast)
    _, ast, _, _, _ = ir.grad(ast, set(["x"]), set(["y1", "y2", "y3"]), set())
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("d_x", (), "int32", "output", "cpu"),
                    ("y1", (), "int32", "input", "cpu"),
                    ("d_y1", (), "int32", "inout", "cpu"),
                    ("y2", (), "int32", "input", "cpu"),
                    ("d_y2", (), "int32", "inout", "cpu"),
                    ("y3", (), "int32", "input", "cpu"),
                    ("d_y3", (), "int32", "inout", "cpu")
                   ]) as (x, d_x, y1, d_y1, y2, d_y2, y3, d_y3):
        with ir.VarDef("d_y3.old", (), "int32", "cache", "cpu") as d_y3_old:
            d_y3_old[()] = d_y3[()]
            d_y3[()] = 0
            with ir.VarDef("d_y2.old", (), "int32", "cache", "cpu") as d_y2_old:
                d_y2_old[()] = d_y2[()]
                d_y2[()] = 0
                with ir.VarDef("d_y1.old", (), "int32", "cache",
                               "cpu") as d_y1_old:
                    d_y1_old[()] = d_y1[()]
                    d_y1[()] = 0
                    d_x[()] = 0 + 2 * d_y3_old[()] * x[()] + d_y2_old[()] * y2[
                        ()] + d_y1_old[()] / (2 * y1[()])
    std = ir.pop_ast()

    assert std.match(ast)


def test_multiple_statements():
    with ir.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("x3", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x1, x2, x3, y1,
                                                               y2):
        with ir.VarDef("t", (), "int32", "cache", "cpu") as t:
            t[()] = x1[()] + x2[()]
            y1[()] = t[()] * x3[()]
            y2[()] = t[()] + x3[()]
    ast = ir.pop_ast()
    print(ast)
    _, ast, _, _, _ = ir.grad(ast, set(["x1", "x2", "x3"]), set(["y1", "y2"]),
                              set())
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("d_x1", (), "int32", "output", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("d_x2", (), "int32", "output", "cpu"),
                    ("x3", (), "int32", "input", "cpu"),
                    ("d_x3", (), "int32", "output", "cpu"),
                    ("y1", (), "int32", "input", "cpu"),
                    ("d_y1", (), "int32", "inout", "cpu"),
                    ("y2", (), "int32", "input", "cpu"),
                    ("d_y2", (), "int32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y1, d_y1, y2, d_y2):
        with ir.VarDef("d_y2.old", (), "int32", "cache", "cpu") as d_y2_old:
            d_y2_old[()] = d_y2[()]
            d_y2[()] = 0
            with ir.VarDef("d_t", (), "int32", "cache", "cpu") as d_t:
                with ir.VarDef("d_y1.old", (), "int32", "cache",
                               "cpu") as d_y1_old:
                    d_y1_old[()] = d_y1[()]
                    d_y1[()] = 0
                    d_x3[()] = d_y2_old[()] + d_y1_old[()] * (x1[()] + x2[()])
                    d_t[()] = d_y2_old[()] + d_y1_old[()] * x3[()]
                d_x1[()] = d_t[()]
                d_x2[()] = d_t[()]
    std = ir.pop_ast()

    assert std.match(ast)


def test_dependent_iterations():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[()] = -y[()] + x[i]
    ast = ir.pop_ast()
    print(ast)
    _, ast, _, _, _ = ir.grad(ast, set(["x"]), set(["y"]), set())
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("d_x", (4,), "int32", "output", "cpu"),
                    ("y", (), "int32", "input", "cpu"),
                    ("d_y", (), "int32", "inout", "cpu")]) as (x, d_x, y, d_y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("d_y.old", (), "int32", "cache", "cpu") as d_y_old:
                d_y_old[()] = d_y[()]
                d_y[()] = -1 * d_y_old[()]
                d_x[-1 * i + 3] = d_y_old[()]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_tape_1():
    with ir.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("x3", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x1, x2, x3, y):
        ir.MarkNid("V_t")
        with ir.VarDef("t", (), "int32", "cache", "cpu") as t:
            t[()] = x1[()] + x2[()]
            y[()] = t[()] * x3[()]
    ast = ir.pop_ast()
    print(ast)
    forward, backward, _, _, _ = ir.grad(ast, set(["x1", "x2", "x3"]),
                                         set(["y"]), set(["V_t"]))
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ir.lower(forward)
    backward = ir.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ir.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("d_x1", (), "int32", "output", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("d_x2", (), "int32", "output", "cpu"),
                    ("x3", (), "int32", "input", "cpu"),
                    ("d_x3", (), "int32", "output", "cpu"),
                    ("y", (), "int32", "input", "cpu"),
                    ("d_y", (), "int32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        with ir.VarDef("t", (), "int32", "input", "cpu") as t:
            with ir.VarDef("d_y.old", (), "int32", "cache", "cpu") as d_y_old:
                d_y_old[()] = d_y[()]
                d_y[()] = 0
                d_x3[()] = 0 + d_y_old[()] * t[()]
                with ir.VarDef("d_t", (), "int32", "cache", "cpu") as d_t:
                    d_t[()] = 0 + d_y_old[()] * x3[()]
                    d_x1[()] = d_t[()]
                    d_x2[()] = d_t[()]
    std = ir.pop_ast()

    assert std.match(backward)


def test_tape_2():
    with ir.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("x3", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x1, x2, x3, y):
        with ir.For("i", 0, 4) as i:
            ir.MarkNid("V_t")
            with ir.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[()] = x1[i] + x2[i]
                y[i] = t[()] * x3[i]
    ast = ir.pop_ast()
    print(ast)
    forward, backward, _, _, _ = ir.grad(ast, set(["x1", "x2", "x3"]),
                                         set(["y"]), set(["V_t"]))
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ir.lower(forward)
    backward = ir.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ir.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("d_x1", (4,), "int32", "output", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("d_x2", (4,), "int32", "output", "cpu"),
                    ("x3", (4,), "int32", "input", "cpu"),
                    ("d_x3", (4,), "int32", "output", "cpu"),
                    ("y", (4,), "int32", "input", "cpu"),
                    ("d_y", (4,), "int32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("t.tape", (4,), "int32", "input", "cpu") as t:
                with ir.VarDef("d_y.old", (), "int32", "cache",
                               "cpu") as d_y_old:
                    d_y_old[()] = d_y[-1 * i + 3]
                    d_y[-1 * i + 3] = 0
                    d_x3[-1 * i + 3] = 0 + d_y_old[()] * t[-1 * i + 3]
                    with ir.VarDef("d_t", (), "int32", "cache", "cpu") as d_t:
                        d_t[()] = 0 + d_y_old[()] * x3[-1 * i + 3]
                        d_x1[-1 * i + 3] = d_t[()]
                        d_x2[-1 * i + 3] = d_t[()]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(backward)


def test_tape_mode_all():
    with ir.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("x3", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x1, x2, x3, y):
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x1[i] + x2[i]
            with ir.For("i", 0, 4) as i:
                with ir.VarDef("u", (), "int32", "cache", "cpu") as u:
                    u[()] = x2[i] + x3[i]
                    y[i] = u[()] * t[i]
    ast = ir.pop_ast()
    print(ast)
    forward, backward, _, _, _ = ir.grad(ast, set(["x1", "x2", "x3"]),
                                         set(["y"]), ir.GradTapeMode.All)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ir.lower(forward)
    backward = ir.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ir.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("d_x1", (4,), "int32", "output", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("d_x2", (4,), "int32", "output", "cpu"),
                    ("x3", (4,), "int32", "input", "cpu"),
                    ("d_x3", (4,), "int32", "output", "cpu"),
                    ("y", (4,), "int32", "input", "cpu"),
                    ("d_y", (4,), "int32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        with ir.VarDef([("t.tape", (4,), "int32", "input", "cpu"),
                        ("d_t", (4,), "int32", "cache", "cpu")]) as (t, d_t):
            with ir.For("i", 0, 4) as i:
                with ir.VarDef("u.tape", (4,), "int32", "input", "cpu") as u:
                    with ir.VarDef("d_y.old", (), "int32", "cache",
                                   "cpu") as d_y_old:
                        d_y_old[()] = d_y[-1 * i + 3]
                        d_y[-1 * i + 3] = 0
                        d_t[-1 * i + 3] = 0 + d_y_old[()] * u[-1 * i + 3]
                        with ir.VarDef("d_u", (), "int32", "cache",
                                       "cpu") as d_u:
                            d_u[()] = 0 + d_y_old[()] * t[-1 * i + 3]
                            d_x2[-1 * i + 3] = d_u[()]
                            d_x3[-1 * i + 3] = d_u[()]
            with ir.For("i", 0, 4) as i:
                with ir.VarDef("d_t.old", (), "int32", "cache",
                               "cpu") as d_t_old:
                    d_t_old[()] = d_t[-1 * i + 3]
                    d_t[-1 * i + 3] = 0
                    d_x1[-1 * i + 3] = d_t_old[()]
                    d_x2[-1 * i + 3] += d_t_old[()]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(backward)


def test_tape_mode_nothing():
    with ir.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("x3", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x1, x2, x3, y):
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x1[i] + x2[i]
            with ir.For("i", 0, 4) as i:
                with ir.VarDef("u", (), "int32", "cache", "cpu") as u:
                    u[()] = x2[i] + x3[i]
                    y[i] = u[()] * t[i]
    ast = ir.pop_ast()
    print(ast)
    forward, backward, _, _, _ = ir.grad(ast, set(["x1", "x2", "x3"]),
                                         set(["y"]), ir.GradTapeMode.Nothing)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ir.lower(forward)
    backward = ir.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ir.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("d_x1", (4,), "int32", "output", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("d_x2", (4,), "int32", "output", "cpu"),
                    ("x3", (4,), "int32", "input", "cpu"),
                    ("d_x3", (4,), "int32", "output", "cpu"),
                    ("y", (4,), "int32", "input", "cpu"),
                    ("d_y", (4,), "int32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x1[i] + x2[i]
            with ir.VarDef("d_t", (4,), "int32", "cache", "cpu") as d_t:
                with ir.For("i", 0, 4) as i:
                    with ir.VarDef("d_y.old", (), "int32", "cache",
                                   "cpu") as d_y_old:
                        d_y_old[()] = d_y[-1 * i + 3]
                        d_y[-1 * i + 3] = 0
                        d_t[-1 * i + 3] = 0 + d_y_old[
                            ()] * (x2[-1 * i + 3] + x3[-1 * i + 3])
                        with ir.VarDef("d_u", (), "int32", "cache",
                                       "cpu") as d_u:
                            d_u[()] = 0 + d_y_old[()] * t[-1 * i + 3]
                            d_x2[-1 * i + 3] = d_u[()]
                            d_x3[-1 * i + 3] = d_u[()]
                with ir.For("i", 0, 4) as i:
                    with ir.VarDef("d_t.old", (), "int32", "cache",
                                   "cpu") as d_t_old:
                        d_t_old[()] = d_t[-1 * i + 3]
                        d_t[-1 * i + 3] = 0
                        d_x1[-1 * i + 3] = d_t_old[()]
                        d_x2[-1 * i + 3] += d_t_old[()]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(backward)


def test_tape_mode_no_reuse_only():
    with ir.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("x3", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x1, x2, x3, y):
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x1[i] + x2[i]
            with ir.For("i", 0, 4) as i:
                with ir.VarDef("u", (), "int32", "cache", "cpu") as u:
                    u[()] = x2[i] + x3[i]
                    y[i] = u[()] * t[i]
    ast = ir.pop_ast()
    print(ast)
    forward, backward, _, _, _ = ir.grad(ast, set(["x1", "x2",
                                                   "x3"]), set(["y"]),
                                         ir.GradTapeMode.NoReuseOnly)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    forward = ir.lower(forward)
    backward = ir.lower(backward)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)

    with ir.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("d_x1", (4,), "int32", "output", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("d_x2", (4,), "int32", "output", "cpu"),
                    ("x3", (4,), "int32", "input", "cpu"),
                    ("d_x3", (4,), "int32", "output", "cpu"),
                    ("y", (4,), "int32", "input", "cpu"),
                    ("d_y", (4,), "int32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        with ir.VarDef([("t.tape", (4,), "int32", "input", "cpu"),
                        ("d_t", (4,), "int32", "cache", "cpu")]) as (t, d_t):
            with ir.For("i", 0, 4) as i:
                with ir.VarDef("d_y.old", (), "int32", "cache",
                               "cpu") as d_y_old:
                    d_y_old[()] = d_y[-1 * i + 3]
                    d_y[-1 * i + 3] = 0
                    d_t[-1 * i +
                        3] = 0 + d_y_old[()] * (x2[-1 * i + 3] + x3[-1 * i + 3])
                    with ir.VarDef("d_u", (), "int32", "cache", "cpu") as d_u:
                        d_u[()] = 0 + d_y_old[()] * t[-1 * i + 3]
                        d_x2[-1 * i + 3] = d_u[()]
                        d_x3[-1 * i + 3] = d_u[()]
            with ir.For("i", 0, 4) as i:
                with ir.VarDef("d_t.old", (), "int32", "cache",
                               "cpu") as d_t_old:
                    d_t_old[()] = d_t[-1 * i + 3]
                    d_t[-1 * i + 3] = 0
                    d_x1[-1 * i + 3] = d_t_old[()]
                    d_x2[-1 * i + 3] += d_t_old[()]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(backward)
