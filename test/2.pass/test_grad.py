import ir


def test_basic():
    with ir.VarDef([
        ("x1", (), "float32", "input", "cpu"),
        ("x2", (), "float32", "input", "cpu"),
        ("x3", (), "float32", "input", "cpu"),
        ("y", (), "float32", "output", "cpu"),
    ]) as (x1, x2, x3, y):
        y[()] = (x1[()] + x2[()]) * x3[()]
    ast = ir.pop_ast()
    print(ast)
    _, ast, _, _, _ = ir.grad(ast, set(["x1", "x2", "x3"]), set(["y"]), set())
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("d_x1", (), "float32", "output", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("d_x2", (), "float32", "output", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("d_x3", (), "float32", "output", "cpu"),
                    ("y", (), "float32", "input", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        d_x1[()] = d_y[()] * x3[()]
        d_x2[()] = d_y[()] * x3[()]
        d_x3[()] = d_y[()] * (x1[()] + x2[()])
    std = ir.pop_ast()

    assert std.match(ast)


def test_partial_gradient():
    with ir.VarDef([
        ("x1", (), "float32", "input", "cpu"),
        ("x2", (), "float32", "input", "cpu"),
        ("x3", (), "float32", "input", "cpu"),
        ("y", (), "float32", "output", "cpu"),
    ]) as (x1, x2, x3, y):
        y[()] = (x1[()] + x2[()]) * x3[()]
    ast = ir.pop_ast()
    print(ast)
    _, ast, _, _, _ = ir.grad(ast, set(["x1"]), set(["y"]), set())
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("d_x1", (), "float32", "output", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "input", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x1, d_x1, x2,
                                                                 x3, y, d_y):
        d_x1[()] = d_y[()] * x3[()]
    std = ir.pop_ast()

    assert std.match(ast)


def test_branching_exprs():
    with ir.VarDef([
        ("x1", (), "float32", "input", "cpu"),
        ("x2", (), "float32", "input", "cpu"),
        ("y1", (), "float32", "output", "cpu"),
        ("y2", (), "float32", "output", "cpu"),
        ("y3", (), "float32", "output", "cpu"),
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

    with ir.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("d_x1", (), "float32", "output", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("d_x2", (), "float32", "output", "cpu"),
                    ("y1", (), "float32", "input", "cpu"),
                    ("d_y1", (), "float32", "inout", "cpu"),
                    ("y2", (), "float32", "input", "cpu"),
                    ("d_y2", (), "float32", "inout", "cpu"),
                    ("y3", (), "float32", "input", "cpu"),
                    ("d_y3", (), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, y1, d_y1, y2, d_y2, y3, d_y3):
        d_x1[()] = ir.if_then_else(x1[()] > 0, d_y3[()], 0) + ir.if_then_else(
            x1[()] >= x2[()], d_y2[()], 0) + ir.if_then_else(
                x1[()] <= x2[()], d_y1[()], 0)
        d_x2[()] = ir.if_then_else(x1[()] <= 0, d_y3[()], 0) + ir.if_then_else(
            x2[()] > x1[()], d_y2[()], 0) + ir.if_then_else(
                x2[()] < x1[()], d_y1[()], 0)
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_math_funcs():
    with ir.VarDef([
        ("x", (), "float32", "input", "cpu"),
        ("y1", (), "float32", "output", "cpu"),
        ("y2", (), "float32", "output", "cpu"),
        ("y3", (), "float32", "output", "cpu"),
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

    with ir.VarDef([("x", (), "float32", "input", "cpu"),
                    ("d_x", (), "float32", "output", "cpu"),
                    ("y1", (), "float32", "input", "cpu"),
                    ("d_y1", (), "float32", "inout", "cpu"),
                    ("y2", (), "float32", "input", "cpu"),
                    ("d_y2", (), "float32", "inout", "cpu"),
                    ("y3", (), "float32", "input", "cpu"),
                    ("d_y3", (), "float32", "inout", "cpu")
                   ]) as (x, d_x, y1, d_y1, y2, d_y2, y3, d_y3):
        d_x[(
        )] = 2 * d_y3[()] * x[()] + d_y2[()] * y2[()] + d_y1[()] / (2 * y1[()])
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_multiple_statements():
    with ir.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("y1", (), "float32", "output", "cpu"),
                    ("y2", (), "float32", "output", "cpu")]) as (x1, x2, x3, y1,
                                                                 y2):
        with ir.VarDef("t", (), "float32", "cache", "cpu") as t:
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

    with ir.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("d_x1", (), "float32", "output", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("d_x2", (), "float32", "output", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("d_x3", (), "float32", "output", "cpu"),
                    ("y1", (), "float32", "input", "cpu"),
                    ("d_y1", (), "float32", "inout", "cpu"),
                    ("y2", (), "float32", "input", "cpu"),
                    ("d_y2", (), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y1, d_y1, y2, d_y2):
        with ir.VarDef("d_t", (), "float32", "cache", "cpu") as d_t:
            d_t[()] = d_y2[()] + d_y1[()] * x3[()]
            d_x3[()] = d_y2[()] + d_y1[()] * (x1[()] + x2[()])
            d_x1[()] = d_t[()]
            d_x2[()] = d_t[()]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_nested_local_def():
    with ir.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("y1", (), "float32", "output", "cpu"),
                    ("y2", (), "float32", "output", "cpu")]) as (x1, x2, x3, y1,
                                                                 y2):
        with ir.VarDef("t", (), "float32", "cache", "cpu") as t:
            with ir.VarDef("u", (), "float32", "cache", "cpu") as u:
                u[()] = x1[()] + x2[()]
                t[()] = u[()] * x3[()]
                y1[()] = u[()] * x1[()]
            y2[()] = t[()] * x2[()]
    ast = ir.pop_ast()
    print(ast)
    _, ast, _, _, _ = ir.grad(ast, set(["x1", "x2", "x3"]), set(["y1", "y2"]),
                              set())
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("d_x1", (), "float32", "output", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("d_x2", (), "float32", "output", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("d_x3", (), "float32", "output", "cpu"),
                    ("y1", (), "float32", "input", "cpu"),
                    ("d_y1", (), "float32", "inout", "cpu"),
                    ("y2", (), "float32", "input", "cpu"),
                    ("d_y2", (), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y1, d_y1, y2, d_y2):
        with ir.VarDef("d_t", (), "float32", "cache", "cpu") as d_t:
            with ir.VarDef("u", (), "float32", "cache", "cpu") as u:
                u[()] = x1[()] + x2[()]
                d_t[()] = d_y2[()] * x2[()]
                with ir.VarDef("d_u", (), "float32", "cache", "cpu") as d_u:
                    d_u[()] = d_y1[()] * x1[()] + d_t[()] * x3[()]
                    d_x3[()] = d_t[()] * u[()]
                    d_x1[()] = d_y1[()] * u[()] + d_u[()]
                    d_x2[()] = d_y2[()] * (u[()] * x3[()]) + d_u[()]
    std = ir.pop_ast()

    assert std.match(ast)


def test_dependent_iterations():
    with ir.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[()] = -y[()] + x[i]
    ast = ir.pop_ast()
    print(ast)
    _, ast, _, _, _ = ir.grad(ast, set(["x"]), set(["y"]), set())
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("d_x", (4,), "float32", "output", "cpu"),
                    ("y", (), "float32", "input", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x, d_x, y,
                                                                 d_y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("d_y.old", (), "float32", "cache", "cpu") as d_y_old:
                d_y_old[()] = d_y[()]
                d_y[()] = -1 * d_y_old[()]
                d_x[-1 * i + 3] = d_y_old[()]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_assign_quick_path():
    with ir.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[()] = x[i]
    ast = ir.pop_ast()
    print(ast)
    _, ast, _, _, _ = ir.grad(ast, set(["x"]), set(["y"]), set())
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("d_x", (4,), "float32", "output", "cpu"),
                    ("y", (), "float32", "input", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x, d_x, y,
                                                                 d_y):
        with ir.For("i", 0, 4) as i:
            d_x[-1 * i + 3] = d_y[-1 * i + 3]
            d_y[-1 * i + 3] = 0
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_reduce_sum_quick_path():
    with ir.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[()] += x[i]
    ast = ir.pop_ast()
    print(ast)
    _, ast, _, _, _ = ir.grad(ast, set(["x"]), set(["y"]), set())
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("d_x", (4,), "float32", "output", "cpu"),
                    ("y", (), "float32", "input", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x, d_x, y,
                                                                 d_y):
        with ir.For("i", 0, 4) as i:
            d_x[-1 * i + 3] = d_y[-1 * i + 3]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_atypical_loop():
    with ir.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        with ir.For("i", -2, 2) as i:
            y[()] = -y[()] + x[i + 2]
    ast = ir.pop_ast()
    print(ast)
    _, ast, _, _, _ = ir.grad(ast, set(["x"]), set(["y"]), set())
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("d_x", (4,), "float32", "output", "cpu"),
                    ("y", (), "float32", "input", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x, d_x, y,
                                                                 d_y):
        with ir.For("i", -2, 2) as i:
            with ir.VarDef("d_y.old", (), "float32", "cache", "cpu") as d_y_old:
                d_y_old[()] = d_y[()]
                d_y[()] = -1 * d_y_old[()]
                d_x[-1 * i + 1] = d_y_old[()]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_nested_loops():
    with ir.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("w0", (4, 4), "float32", "input", "cpu"),
                    ("w1", (4, 4), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x, w0, w1, y):
        with ir.VarDef("t", (4,), "float32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = 0
                with ir.For("j", 0, 4) as j:
                    t[i] += x[j] * w0[i, j]
            with ir.For("i", 0, 4) as i:
                y[i] = 0
                with ir.For("j", 0, 4) as j:
                    y[i] += t[j] * w1[i, j]
    ast = ir.pop_ast()
    print(ast)
    _, ast, _, _, _ = ir.grad(ast, set(["x", "w0", "w1"]), set(["y"]), set())
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("d_x", (4,), "float32", "output", "cpu"),
                    ("w0", (4, 4), "float32", "input", "cpu"),
                    ("d_w0", (4, 4), "float32", "output", "cpu"),
                    ("w1", (4, 4), "float32", "input", "cpu"),
                    ("d_w1", (4, 4), "float32", "output", "cpu"),
                    ("y", (4,), "float32", "input", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu")
                   ]) as (x, d_x, w0, d_w0, w1, d_w1, y, d_y):
        with ir.For("i0", 0, 4) as i:
            d_x[-1 * i + 3] = 0
        with ir.VarDef("t", (4,), "float32", "cache", "cpu") as t:
            with ir.VarDef("d_t", (4,), "float32", "cache", "cpu") as d_t:
                with ir.For("i1", 0, 4) as i:
                    d_t[-1 * i + 3] = 0
                with ir.For("i", 0, 4) as i:
                    t[i] = 0
                    with ir.For("j", 0, 4) as j:
                        t[i] += x[j] * w0[i, j]
                with ir.For("i", 0, 4) as i:
                    with ir.For("j", 0, 4) as j:
                        d_t[-1 * j +
                            3] += d_y[-1 * i + 3] * w1[-1 * i + 3, -1 * j + 3]
                        d_w1[-1 * i + 3,
                             -1 * j + 3] = d_y[-1 * i + 3] * t[-1 * j + 3]
                    d_y[-1 * i + 3] = 0
                with ir.For("i", 0, 4) as i:
                    with ir.For("j", 0, 4) as j:
                        d_x[-1 * j +
                            3] += d_t[-1 * i + 3] * w0[-1 * i + 3, -1 * j + 3]
                        d_w0[-1 * i + 3,
                             -1 * j + 3] = d_t[-1 * i + 3] * x[-1 * j + 3]
                    d_t[-1 * i + 3] = 0
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_tape_1():
    with ir.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, x3, y):
        ir.MarkNid("V_t")
        with ir.VarDef("t", (), "float32", "cache", "cpu") as t:
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

    with ir.VarDef([("x1", (), "float32", "input", "cpu"),
                    ("d_x1", (), "float32", "output", "cpu"),
                    ("x2", (), "float32", "input", "cpu"),
                    ("d_x2", (), "float32", "output", "cpu"),
                    ("x3", (), "float32", "input", "cpu"),
                    ("d_x3", (), "float32", "output", "cpu"),
                    ("y", (), "float32", "input", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        with ir.VarDef("t", (), "float32", "input", "cpu") as t:
            with ir.VarDef("d_t", (), "float32", "cache", "cpu") as d_t:
                d_t[()] = d_y[()] * x3[()]
                d_x3[()] = d_y[()] * t[()]
                d_x1[()] = d_t[()]
                d_x2[()] = d_t[()]
    std = ir.pop_ast()

    assert std.match(backward)


def test_tape_2():
    with ir.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x1, x2, x3,
                                                                  y):
        with ir.For("i", 0, 4) as i:
            ir.MarkNid("V_t")
            with ir.VarDef("t", (), "float32", "cache", "cpu") as t:
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

    with ir.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("d_x1", (4,), "float32", "output", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("d_x2", (4,), "float32", "output", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("d_x3", (4,), "float32", "output", "cpu"),
                    ("y", (4,), "float32", "input", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("t.tape", (4,), "float32", "input", "cpu") as t:
                with ir.VarDef("d_t", (), "float32", "cache", "cpu") as d_t:
                    d_t[()] = d_y[-1 * i + 3] * x3[-1 * i + 3]
                    d_x3[-1 * i + 3] = d_y[-1 * i + 3] * t[-1 * i + 3]
                    d_x1[-1 * i + 3] = d_t[()]
                    d_x2[-1 * i + 3] = d_t[()]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(backward)


def test_tape_3():
    with ir.VarDef([("x1", (4, 5, 6), "float32", "input", "cpu"),
                    ("x2", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6), "float32", "output", "cpu")]) as (x1, x2, y):
        with ir.For("i", 0, 4, nid="Li") as i:
            ir.MarkNid("V_t")
            with ir.VarDef("t", (6,), "float32", "cache", "cpu") as t:
                with ir.For("k", 0, 6, nid="Lk0") as k:
                    t[k] = 0
                with ir.For("j", 0, 5, nid="Lj") as j:
                    with ir.For("k", 0, 6, nid="Lk1") as k:
                        t[k] += x1[i, j, k]
                with ir.For("k", 0, 6, nid="Lk2"):
                    y[i, k] = 0
                    with ir.For("j", 0, 5, nid="Lj") as j:
                        y[i, k] += t[k] * x2[i, j, k]
    ast = ir.pop_ast()
    print(ast)
    forward, backward, _, _, _ = ir.grad(ast, set(["x2"]), set(["y"]),
                                         set(["V_t"]))
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

    with ir.VarDef([("x1", (4, 5, 6), "float32", "input", "cpu"),
                    ("x2", (4, 5, 6), "float32", "input", "cpu"),
                    ("d_x2", (4, 5, 6), "float32", "output", "cpu"),
                    ("y", (4, 6), "float32", "input", "cpu"),
                    ("d_y", (4, 6), "float32", "inout", "cpu")
                   ]) as (x1, x2, d_x2, y, d_y):
        with ir.For("i", 0, 4, nid="Li") as i:
            with ir.VarDef("t", (4, 6), "float32", "input", "cpu") as t:
                with ir.For("k", 0, 6, nid="Lk2"):
                    with ir.For("j", 0, 5, nid="Lj") as j:
                        d_x2[-1 * i + 3, -1 * j + 4, -1 * k +
                             5] = d_y[-1 * i + 3, -1 * k + 5] * t[-1 * i + 3,
                                                                  -1 * k + 5]
                    d_y[-1 * i + 3, -1 * k + 5] = 0
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(backward)


def test_tape_4():
    with ir.VarDef([("x", (100,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        ir.MarkNid("V_t")
        with ir.VarDef("t", (), "float32", "cache", "cpu") as t:
            t[()] = 1
            with ir.For("i", 0, 100) as i:
                t[()] = t[()] * x[i] + 1
            y[()] = t[()]
    ast = ir.pop_ast()
    print(ast)
    forward, backward, _, _, _ = ir.grad(ast, set(["x"]), set(["y"]),
                                         set(["V_t"]))
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

    with ir.VarDef([("x", (100,), "float32", "input", "cpu"),
                    ("d_x", (100,), "float32", "output", "cpu"),
                    ("y", (), "float32", "input", "cpu"),
                    ("d_y", (), "float32", "inout", "cpu")]) as (x, d_x, y,
                                                                 d_y):
        with ir.VarDef([("t", (101,), "float32", "input", "cpu"),
                        ("d_t", (), "float32", "cache", "cpu")]) as (t, d_t):
            d_t[()] = d_y[()]
            with ir.For("i", 0, 100) as i:
                with ir.VarDef("d_t_old", (), "float32", "cache",
                               "cpu") as d_t_old:
                    d_t_old[()] = d_t[()]
                    d_t[()] = d_t_old[()] * x[-1 * i + 99]
                    d_x[-1 * i + 99] = d_t_old[()] * t[-1 * i + 99]
            d_t[()] = 0
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(backward)


def test_tape_mode_all():
    with ir.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x1, x2, x3,
                                                                  y):
        with ir.VarDef("t", (4,), "float32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x1[i] + x2[i]
            with ir.For("i", 0, 4) as i:
                with ir.VarDef("u", (), "float32", "cache", "cpu") as u:
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

    with ir.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("d_x1", (4,), "float32", "output", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("d_x2", (4,), "float32", "output", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("d_x3", (4,), "float32", "output", "cpu"),
                    ("y", (4,), "float32", "input", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        with ir.VarDef([("t.tape", (4,), "float32", "input", "cpu"),
                        ("d_t", (4,), "float32", "cache", "cpu")]) as (t, d_t):
            with ir.For("i", 0, 4) as i:
                with ir.VarDef("u.tape", (4,), "float32", "input", "cpu") as u:
                    with ir.VarDef("d_u", (), "float32", "cache", "cpu") as d_u:
                        d_u[()] = d_y[-1 * i + 3] * t[-1 * i + 3]
                        d_t[-1 * i + 3] = d_y[-1 * i + 3] * u[-1 * i + 3]
                        d_x2[-1 * i + 3] = d_u[()]
                        d_x3[-1 * i + 3] = d_u[()]
            with ir.For("i", 0, 4) as i:
                d_x1[-1 * i + 3] = d_t[-1 * i + 3]
                d_x2[-1 * i + 3] += d_t[-1 * i + 3]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(backward)


def test_tape_mode_nothing():
    with ir.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x1, x2, x3,
                                                                  y):
        with ir.VarDef("t", (4,), "float32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x1[i] + x2[i]
            with ir.For("i", 0, 4) as i:
                with ir.VarDef("u", (), "float32", "cache", "cpu") as u:
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

    with ir.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("d_x1", (4,), "float32", "output", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("d_x2", (4,), "float32", "output", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("d_x3", (4,), "float32", "output", "cpu"),
                    ("y", (4,), "float32", "input", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        with ir.VarDef("t", (4,), "float32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x1[i] + x2[i]
            with ir.VarDef("d_t", (4,), "float32", "cache", "cpu") as d_t:
                with ir.For("i", 0, 4) as i:
                    with ir.VarDef("d_u", (), "float32", "cache", "cpu") as d_u:
                        d_u[()] = d_y[-1 * i + 3] * t[-1 * i + 3]
                        d_t[-1 * i + 3] = d_y[-1 * i + 3] * (x2[-1 * i + 3] +
                                                             x3[-1 * i + 3])
                        d_x2[-1 * i + 3] = d_u[()]
                        d_x3[-1 * i + 3] = d_u[()]
                with ir.For("i", 0, 4) as i:
                    d_x1[-1 * i + 3] = d_t[-1 * i + 3]
                    d_x2[-1 * i + 3] += d_t[-1 * i + 3]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(backward)


def test_tape_mode_no_reuse_only():
    with ir.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("y", (4,), "float32", "output", "cpu")]) as (x1, x2, x3,
                                                                  y):
        with ir.VarDef("t", (4,), "float32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x1[i] + x2[i]
            with ir.For("i", 0, 4) as i:
                with ir.VarDef("u", (), "float32", "cache", "cpu") as u:
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

    with ir.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("d_x1", (4,), "float32", "output", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("d_x2", (4,), "float32", "output", "cpu"),
                    ("x3", (4,), "float32", "input", "cpu"),
                    ("d_x3", (4,), "float32", "output", "cpu"),
                    ("y", (4,), "float32", "input", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu")
                   ]) as (x1, d_x1, x2, d_x2, x3, d_x3, y, d_y):
        with ir.VarDef([("t.tape", (4,), "float32", "input", "cpu"),
                        ("d_t", (4,), "float32", "cache", "cpu")]) as (t, d_t):
            with ir.For("i", 0, 4) as i:
                with ir.VarDef("d_u", (), "float32", "cache", "cpu") as d_u:
                    d_u[()] = d_y[-1 * i + 3] * t[-1 * i + 3]
                    d_t[-1 * i +
                        3] = d_y[-1 * i + 3] * (x2[-1 * i + 3] + x3[-1 * i + 3])
                    d_x2[-1 * i + 3] = d_u[()]
                    d_x3[-1 * i + 3] = d_u[()]
            with ir.For("i", 0, 4) as i:
                d_x1[-1 * i + 3] = d_t[-1 * i + 3]
                d_x2[-1 * i + 3] += d_t[-1 * i + 3]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(backward)


def test_no_deps():

    @ir.transform
    def test(ptr, edge1, edge2):
        ir.declare_var(ptr, (11,), "int32", "input", "cpu")
        ir.declare_var(edge1, (50,), "float32", "input", "cpu")
        ir.declare_var(edge2, (50,), "float32", "output", "cpu")
        'nid: Li'
        'no_deps: edge2'
        for i in range(10):
            for j in range(ptr[i], ptr[i + 1]):
                edge2[j] = edge1[j] + i

    print(test)
    _, backward, _, _, _ = ir.grad(test, set(["edge1"]), set(["edge2"]), set())
    print(backward)
    s = ir.Schedule(backward)
    s.parallelize("Li", "openmp")  # No exception here
    print(s.ast())
