import freetensor as ft


def test_reuse_over_loop():
    with ft.VarDef([("x", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="Li") as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (6,), "float32", "cache", "cpu") as t:
                with ft.For("k", 0, 6, label="Lk0") as k:
                    t[k] = 0
                with ft.For("j", 0, 5, label="Lj") as j:
                    with ft.For("k", 0, 6, label="Lk1") as k:
                        t[k] += x[i, j, k]
                with ft.For("k", 0, 6, label="Lk2"):
                    y[i, k] = t[k] * 2
    ast = ft.pop_ast()
    print(ast)
    ast = ft.output_intermediates(ast, set(["V_t"]))
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t.tape", (4, 6), "float32", "output",
                           "cpu") as t_tape:
                with ft.VarDef("t", (6,), "float32", "cache", "cpu") as t:
                    with ft.For("k", 0, 6) as k:
                        t[k] = 0
                    with ft.For("j", 0, 5) as j:
                        with ft.For("k", 0, 6) as k:
                            t[k] += x[i, j, k]
                    with ft.For("k", 0, 6):
                        t_tape[i, k] = t[k]
                        y[i, k] = t[k] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_multiple_assignments():
    with ft.VarDef([("x", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6, 2), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="Li") as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (6, 2), "float32", "cache", "cpu") as t:
                with ft.For("k", 0, 6, label="Lk0") as k:
                    t[k, 0] = 0
                    t[k, 1] = 0
                with ft.For("j", 0, 5, label="Lj") as j:
                    with ft.For("k", 0, 6, label="Lk1") as k:
                        t[k, 0] += x[i, j, k]
                        t[k, 1] += x[i, j, k] + 1
                with ft.For("k", 0, 6, label="Lk2"):
                    y[i, k, 0] = t[k, 0] * 2
                    y[i, k, 1] = t[k, 1] * 2
    ast = ft.pop_ast()
    print(ast)
    ast = ft.output_intermediates(ast, set(["V_t"]))
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6, 2), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t.tape", (4, 6, 2), "float32", "output",
                           "cpu") as t_tape:
                with ft.VarDef("t", (6, 2), "float32", "cache", "cpu") as t:
                    with ft.For("k", 0, 6) as k:
                        t[k, 0] = 0
                        t[k, 1] = 0
                    with ft.For("j", 0, 5) as j:
                        with ft.For("k", 0, 6) as k:
                            t[k, 0] += x[i, j, k]
                            t[k, 1] += x[i, j, k] + 1
                    with ft.For("k", 0, 6):
                        t_tape[i, k, 0] = t[k, 0]
                        y[i, k, 0] = t[k, 0] * 2
                        t_tape[i, k, 1] = t[k, 1]
                        y[i, k, 1] = t[k, 1] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_reuse_over_loop_with_offset():
    with ft.VarDef([("x", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 2, 6, label="Li") as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (6,), "float32", "cache", "cpu") as t:
                with ft.For("k", 0, 6, label="Lk0") as k:
                    t[k] = 0
                with ft.For("j", 0, 5, label="Lj") as j:
                    with ft.For("k", 0, 6, label="Lk1") as k:
                        t[k] += x[i - 2, j, k]
                with ft.For("k", 0, 6, label="Lk2"):
                    y[i - 2, k] = t[k] * 2
    ast = ft.pop_ast()
    print(ast)
    ast = ft.output_intermediates(ast, set(["V_t"]))
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 2, 6) as i:
            with ft.VarDef("t.tape", (4, 6), "float32", "output",
                           "cpu") as t_tape:
                with ft.VarDef("t", (6,), "float32", "cache", "cpu") as t:
                    with ft.For("k", 0, 6) as k:
                        t[k] = 0
                    with ft.For("j", 0, 5) as j:
                        with ft.For("k", 0, 6) as k:
                            t[k] += x[i + -2, j, k]
                    with ft.For("k", 0, 6):
                        t_tape[i + -2, k] = t[k]
                        y[i + -2, k] = t[k] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_reuse_over_stmt_seq():
    with ft.VarDef([("x", (6,), "float32", "input", "cpu"),
                    ("y", (6,), "float32", "inout", "cpu")]) as (x, y):
        ft.MarkLabel("V_t")
        with ft.VarDef("t", (6,), "float32", "cache", "cpu") as t:
            with ft.For("k", 0, 6, label="Lk0") as k:
                t[k] = x[k] * k
            with ft.For("k", 0, 6, label="Lk2"):
                y[k] += t[k] * 2
            with ft.For("k", 0, 6, label="Lk0") as k:
                t[k] = x[5 - k] * k
            with ft.For("k", 0, 6, label="Lk2"):
                y[k] *= t[k]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.output_intermediates(ast, set(["V_t"]))
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (6,), "float32", "input", "cpu"),
                    ("y", (6,), "float32", "inout", "cpu")]) as (x, y):
        with ft.VarDef("t.tape", (2, 6), "float32", "output", "cpu") as t_tape:
            with ft.VarDef("t", (6,), "float32", "cache", "cpu") as t:
                with ft.For("k", 0, 6) as k:
                    t[k] = x[k] * k
                with ft.For("k", 0, 6):
                    t_tape[0, k] = t[k]
                    y[k] += t[k] * 2
                with ft.For("k", 0, 6) as k:
                    t[k] = x[-1 * k + 5] * k
                with ft.For("k", 0, 6):
                    t_tape[1, k] = t[k]
                    y[k] *= t[k]
    std = ft.pop_ast()

    assert std.match(ast)


def test_reuse_different_lengths():
    with ft.VarDef([("x1", (4, 5, 6), "float32", "input", "cpu"),
                    ("x2", (2, 5, 6), "float32", "input", "cpu"),
                    ("y1", (4, 6), "float32", "output", "cpu"),
                    ("y2", (2, 6), "float32", "output", "cpu")]) as (x1, x2, y1,
                                                                     y2):
        ft.MarkLabel("V_t")
        with ft.VarDef("t", (6,), "float32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                with ft.For("k", 0, 6) as k:
                    t[k] = 0
                with ft.For("j", 0, 5) as j:
                    with ft.For("k", 0, 6) as k:
                        t[k] += x1[i, j, k]
                with ft.For("k", 0, 6):
                    y1[i, k] = t[k] * 2
            with ft.For("i", 0, 2) as i:
                with ft.For("k", 0, 6) as k:
                    t[k] = 0
                with ft.For("j", 0, 5) as j:
                    with ft.For("k", 0, 6) as k:
                        t[k] += x2[i, j, k]
                with ft.For("k", 0, 6):
                    y2[i, k] = t[k] * 2
    ast = ft.pop_ast()
    print(ast)
    ast = ft.output_intermediates(ast, set(["V_t"]))
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (4, 5, 6), "float32", "input", "cpu"),
                    ("x2", (2, 5, 6), "float32", "input", "cpu"),
                    ("y1", (4, 6), "float32", "output", "cpu"),
                    ("y2", (2, 6), "float32", "output", "cpu")]) as (x1, x2, y1,
                                                                     y2):
        with ft.VarDef("t.tape", (6, 6), "float32", "output", "cpu") as t_tape:
            with ft.VarDef("t", (6,), "float32", "cache", "cpu") as t:
                with ft.For("i", 0, 4) as i:
                    with ft.For("k", 0, 6) as k:
                        t[k] = 0
                    with ft.For("j", 0, 5) as j:
                        with ft.For("k", 0, 6) as k:
                            t[k] += x1[i, j, k]
                    with ft.For("k", 0, 6):
                        t_tape[i, k] = t[k]
                        y1[i, k] = t[k] * 2
                with ft.For("i", 0, 2) as i:
                    with ft.For("k", 0, 6) as k:
                        t[k] = 0
                    with ft.For("j", 0, 5) as j:
                        with ft.For("k", 0, 6) as k:
                            t[k] += x2[i, j, k]
                    with ft.For("k", 0, 6):
                        t_tape[4 + i, k] = t[k]
                        y2[i, k] = t[k] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_no_need_to_copy():
    with ft.VarDef([("x", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="Li") as i:
            ft.MarkLabel("V_t")
            with ft.VarDef("t", (4, 6), "float32", "cache", "cpu") as t:
                with ft.For("k", 0, 6, label="Lk0") as k:
                    t[i, k] = 0
                with ft.For("j", 0, 5, label="Lj") as j:
                    with ft.For("k", 0, 6, label="Lk1") as k:
                        t[i, k] += x[i, j, k]
                with ft.For("k", 0, 6, label="Lk2"):
                    y[i, k] = t[i, k] * 2
    ast = ft.pop_ast()
    print(ast)
    ast = ft.output_intermediates(ast, set(["V_t"]))
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4, 5, 6), "float32", "input", "cpu"),
                    ("y", (4, 6), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (4, 6), "float32", "output", "cpu") as t:
                with ft.For("k", 0, 6) as k:
                    t[i, k] = 0
                with ft.For("j", 0, 5) as j:
                    with ft.For("k", 0, 6) as k:
                        t[i, k] += x[i, j, k]
                with ft.For("k", 0, 6):
                    y[i, k] = t[i, k] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_circular_reuse():
    with ft.VarDef("y", (128,), "float32", "output", "cpu") as y:
        ft.MarkLabel("V_c")
        with ft.VarDef("c", (128,), "float32", "cache", "cpu") as c:
            ft.MarkLabel("V_h")
            with ft.VarDef("h", (128,), "float32", "cache", "cpu") as h:
                with ft.For("i", 0, 128, label='Li0') as i:
                    c[i] = 0
                    h[i] = 0
                with ft.For("p", 0, 100, label='Lp') as p:
                    with ft.For("i", 0, 128, label='Li1') as i:
                        c[i] = h[i] / 2 - 1
                        h[i] = c[i] * 2 + 1
                with ft.For("i", 0, 128, label='Li2') as i:
                    y[i] = h[i]
    ast = ft.pop_ast()
    print(ast)
    ast = ft.output_intermediates(ast, set(["V_c", "V_h"]))
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("y", (128,), "float32", "output", "cpu") as y:
        with ft.VarDef("c.tape", (100, 128), "float32", "output",
                       "cpu") as c_tape:
            with ft.VarDef("h.tape", (101, 128), "float32", "output",
                           "cpu") as h_tape:
                ft.MarkLabel("V_h")
                with ft.VarDef("h", (128,), "float32", "cache", "cpu") as h:
                    with ft.For("i", 0, 128, label='Li0') as i:
                        h[i] = 0
                    ft.MarkLabel("V_c")
                    with ft.VarDef("c", (128,), "float32", "cache", "cpu") as c:
                        with ft.For("p", 0, 100, label='Lp') as p:
                            with ft.For("i", 0, 128, label='Li1') as i:
                                h_tape[p, i] = h[i]
                                c[i] = h[i] / 2 - 1
                                c_tape[p, i] = c[i]
                                h[i] = c[i] * 2 + 1
                    with ft.For("i", 0, 128, label='Li2') as i:
                        h_tape[100, i] = h[i]
                        y[i] = h[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_dynamic_loop_range():

    @ft.transform(verbose=1)
    def func(n: ft.Var[(), "int32"], x, y):
        x: ft.Var[(100, n), "float32"]
        y: ft.Var[(n,), "float32", "output"]

        for bn in range(100):
            for pn in range(n * n):
                #! label: V_z
                z = ft.empty((), "float32")
                z[()] = x[bn, pn] + 1
                y[pn % n] += z[()] * z[()]

    ast = ft.output_intermediates(func.body, set(["V_z"]))
    print(ast)
    ast = ft.lower(ast, skip_passes=['float_simplify'], verbose=1)

    @ft.transform
    def expected(n: ft.Var[(), "int32"], x, y, z_tape):
        x: ft.Var[(100, n), "float32"]
        y: ft.Var[(n,), "float32", "output"]

        for bn in range(100):
            for pn in range(n * n):
                z_tape: ft.Var[(100 * n * n,), "float32", "output"]
                #! label: V_z
                z = ft.empty((), "float32")
                z[()] = x[bn, pn] + 1
                # FIXME: Apparently there 3 assignments are the same, but pass/remove_writes can't
                # eliminate 2 of them, because `bn * (n * n)` and `n * n` are recoginized two
                # different free variables for isl. Consider directly match the expressions
                z_tape[bn * (n * n) + pn] = z[()]
                z_tape[bn * (n * n) + pn + 1 - 1] = z[()]
                z_tape[bn * (n * n) + pn + 1 - 1] = z[()]
                y[pn % n] += z[()] * z[()]

    assert expected.body.match(ast)
