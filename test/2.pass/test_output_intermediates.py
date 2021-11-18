import ir


def test_reuse_over_loop():
    with ir.VarDef([("x", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6), "float32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="Li") as i:
            ir.MarkNid("V_t")
            with ir.VarDef("t", (6,), "float32", "cache", "cpu") as t:
                with ir.For("k", 0, 6, nid="Lk0") as k:
                    t[k] = 0
                with ir.For("j", 0, 5, nid="Lj") as j:
                    with ir.For("k", 0, 6, nid="Lk1") as k:
                        t[k] += x[i, j, k]
                with ir.For("k", 0, 6, nid="Lk2"):
                    y[i, k] = t[k] * 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.output_intermediates(ast, set(["V_t"]))
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6), "float32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("t.tape", (4, 6), "float32", "output",
                           "cpu") as t_tape:
                with ir.VarDef("t", (6,), "float32", "cache", "cpu") as t:
                    with ir.For("k", 0, 6) as k:
                        t[k] = 0
                    with ir.For("j", 0, 5) as j:
                        with ir.For("k", 0, 6) as k:
                            t[k] += x[i, j, k]
                            with ir.If(j == 4):
                                t_tape[i, k] = t[k]
                    with ir.For("k", 0, 6):
                        y[i, k] = t[k] * 2
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_reuse_over_stmt_seq():
    with ir.VarDef([("x", (6,), "float32", "input", "cpu"),
                    ("y", (6,), "float32", "inout", "cpu")]) as (x, y):
        ir.MarkNid("V_t")
        with ir.VarDef("t", (6,), "float32", "cache", "cpu") as t:
            with ir.For("k", 0, 6, nid="Lk0") as k:
                t[k] = x[k] * k
            with ir.For("k", 0, 6, nid="Lk2"):
                y[k] += t[k] * 2
            with ir.For("k", 0, 6, nid="Lk0") as k:
                t[k] = x[5 - k] * k
            with ir.For("k", 0, 6, nid="Lk2"):
                y[k] *= t[k]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.output_intermediates(ast, set(["V_t"]))
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (6,), "float32", "input", "cpu"),
                    ("y", (6,), "float32", "inout", "cpu")]) as (x, y):
        with ir.VarDef("t.tape", (2, 6), "float32", "output", "cpu") as t_tape:
            with ir.VarDef("t", (6,), "float32", "cache", "cpu") as t:
                with ir.For("k", 0, 6) as k:
                    t[k] = x[k] * k
                    t_tape[0, k] = t[k]
                with ir.For("k", 0, 6):
                    y[k] += t[k] * 2
                with ir.For("k", 0, 6) as k:
                    t[k] = x[-1 * k + 5] * k
                    t_tape[1, k] = t[k]
                with ir.For("k", 0, 6):
                    y[k] *= t[k]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_reuse_different_lengths():
    with ir.VarDef([("x1", (4, 5, 6), "float32", "input", "cpu"),
                    ("x2", (2, 5, 6), "float32", "input", "cpu"),
                    ("y1", (4, 6), "float32", "output", "cpu"),
                    ("y2", (2, 6), "float32", "output", "cpu")]) as (x1, x2, y1,
                                                                     y2):
        ir.MarkNid("V_t")
        with ir.VarDef("t", (6,), "float32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                with ir.For("k", 0, 6) as k:
                    t[k] = 0
                with ir.For("j", 0, 5) as j:
                    with ir.For("k", 0, 6) as k:
                        t[k] += x1[i, j, k]
                with ir.For("k", 0, 6):
                    y1[i, k] = t[k] * 2
            with ir.For("i", 0, 2) as i:
                with ir.For("k", 0, 6) as k:
                    t[k] = 0
                with ir.For("j", 0, 5) as j:
                    with ir.For("k", 0, 6) as k:
                        t[k] += x2[i, j, k]
                with ir.For("k", 0, 6):
                    y2[i, k] = t[k] * 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.output_intermediates(ast, set(["V_t"]))
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x1", (4, 5, 6), "float32", "input", "cpu"),
                    ("x2", (2, 5, 6), "float32", "input", "cpu"),
                    ("y1", (4, 6), "float32", "output", "cpu"),
                    ("y2", (2, 6), "float32", "output", "cpu")]) as (x1, x2, y1,
                                                                     y2):
        with ir.VarDef("t.tape", (6, 6), "float32", "output", "cpu") as t_tape:
            with ir.VarDef("t", (6,), "float32", "cache", "cpu") as t:
                with ir.For("i", 0, 4) as i:
                    with ir.For("k", 0, 6) as k:
                        t[k] = 0
                    with ir.For("j", 0, 5) as j:
                        with ir.For("k", 0, 6) as k:
                            t[k] += x1[i, j, k]
                            with ir.If(j == 4):
                                t_tape[i, k] = t[k]
                    with ir.For("k", 0, 6):
                        y1[i, k] = t[k] * 2
                with ir.For("i", 0, 2) as i:
                    with ir.For("k", 0, 6) as k:
                        t[k] = 0
                    with ir.For("j", 0, 5) as j:
                        with ir.For("k", 0, 6) as k:
                            t[k] += x2[i, j, k]
                            with ir.If(j == 4):
                                t_tape[4 + i, k] = t[k]
                    with ir.For("k", 0, 6):
                        y2[i, k] = t[k] * 2
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_no_need_to_copy():
    with ir.VarDef([("x", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6), "float32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="Li") as i:
            ir.MarkNid("V_t")
            with ir.VarDef("t", (4, 6), "float32", "cache", "cpu") as t:
                with ir.For("k", 0, 6, nid="Lk0") as k:
                    t[i, k] = 0
                with ir.For("j", 0, 5, nid="Lj") as j:
                    with ir.For("k", 0, 6, nid="Lk1") as k:
                        t[i, k] += x[i, j, k]
                with ir.For("k", 0, 6, nid="Lk2"):
                    y[i, k] = t[i, k] * 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.output_intermediates(ast, set(["V_t"]))
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6), "float32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("t", (4, 6), "float32", "output", "cpu") as t:
                with ir.For("k", 0, 6) as k:
                    t[i, k] = 0
                with ir.For("j", 0, 5) as j:
                    with ir.For("k", 0, 6) as k:
                        t[i, k] += x[i, j, k]
                with ir.For("k", 0, 6):
                    y[i, k] = t[i, k] * 2
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_circular_reuse():
    with ir.VarDef("y", (128,), "float32", "output", "cpu") as y:
        ir.MarkNid("V_c")
        with ir.VarDef("c", (128,), "float32", "cache", "cpu") as c:
            ir.MarkNid("V_h")
            with ir.VarDef("h", (128,), "float32", "cache", "cpu") as h:
                with ir.For("i", 0, 128, nid='Li0') as i:
                    c[i] = 0
                    h[i] = 0
                with ir.For("p", 0, 100, nid='Lp') as p:
                    with ir.For("i", 0, 128, nid='Li1') as i:
                        c[i] = h[i] / 2 - 1
                        h[i] = c[i] * 2 + 1
                with ir.For("i", 0, 128, nid='Li2') as i:
                    y[i] = h[i]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.output_intermediates(ast, set(["V_c", "V_h"]))
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (128,), "float32", "output", "cpu") as y:
        with ir.VarDef("c.tape", (100, 128), "float32", "output",
                       "cpu") as c_tape:
            ir.MarkNid("V_c")
            with ir.VarDef("c", (128,), "float32", "cache", "cpu") as c:
                with ir.VarDef("h.tape", (101, 128), "float32", "output",
                               "cpu") as h_tape:
                    ir.MarkNid("V_h")
                    with ir.VarDef("h", (128,), "float32", "cache", "cpu") as h:
                        with ir.For("i", 0, 128, nid='Li0') as i:
                            h[i] = 0
                            h_tape[0, i] = 0
                        with ir.For("p", 0, 100, nid='Lp') as p:
                            with ir.For("i", 0, 128, nid='Li1') as i:
                                c[i] = h[i] / 2 - 1
                                c_tape[p, i] = c[i]
                                h[i] = c[i] * 2 + 1
                                h_tape[1 + p, i] = h[i]
                        with ir.For("i", 0, 128, nid='Li2') as i:
                            y[i] = h[i]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)
