import freetensor as ft


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
                                              ["V_t"],
                                              reset_provided_grad=False)
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
        ("d_y", (), "float32", "input", "cpu"),
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
                                              ["V_t"],
                                              reset_provided_grad=False)
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
                    ("d_y", (4,), "float32", "input", "cpu")
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
    forward, backward, _, _, _ = ft.grad_body(ast, ["x2"], ["y"], ["V_t"],
                                              invert=False)
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
    forward, backward, _, _, _ = ft.grad_body(ast, ["x"], ["y"], ["V_t"],
                                              reset_provided_grad=False)
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
                    ("d_y", (), "float32", "input", "cpu")]) as (x, d_x, d_y):
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
    with ft.VarDef([("y", (256,), "float32", "output", "cpu"),
                    ("u", (256, 256), "float32", "input", "cpu")]) as (y, u):
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
    forward, backward, _, _, _ = ft.grad_body(ast, ["u"], ["y"], ["h", "f"],
                                              reset_provided_grad=False)
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

    with ft.VarDef([("y.grad", (256,), "float32", "input", "cpu"),
                    ("u", (256, 256), "float32", "input", "cpu"),
                    ("u.grad", (256, 256), "float32", "output", "cpu"),
                    ("h.tape", (101, 256), "float32", "input", "cpu")
                   ]) as (dy, u, du, h_tape):
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
                                              ["V_t"],
                                              reset_provided_grad=False)
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
        ("d_y", (4, 2, 4), "float32", "input", "cpu"),
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
    forward, backward, _, _ = ft.grad_(func, ["x1", "x2", "x3"], ["y"], ["V_t"],
                                       reset_provided_grad=False)
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
                    ("d_y", (4,), "float32", "input", "cpu")
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
                                       reset_provided_grad=False,
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
        dy: ft.Var[(3,), "float32", "input"]
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
    func = ft.Func("main", ["x", "w"], [], ft.pop_ast())
    print(func)
    forward, backward, _, _ = ft.grad_(func, ["x"], ["w"], ["V_y"],
                                       reset_provided_grad=False)
    print("Forward:")
    print(forward)
    print("Backward:")
    print(backward)
    backward = ft.lower(backward)
    print("Backward:")
    print(backward)

    with ft.VarDef([("x", (), "float32", "input", "cpu"),
                    ("x.grad", (), "float32", "output", "cpu"),
                    ("w.grad", (), "float32", "input", "cpu")]) as (x, dx, dw):
        with ft.VarDef("y", (), "float32>=0", "input", "cpu") as y:
            dx[...] = 0
            with ft.If(x[...] > 0):
                dx[...] = 8 * y[...] * dw[...] * ft.square(y[...]) * x[...]
    std = ft.pop_ast()

    assert std.match(backward.body)


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
    def expected(a, d_a, b_tape, d_b, d_c):
        a: ft.Var[(10,), "float32", "input"]
        d_a: ft.Var[(10,), "float32", "output"]
        b_tape: ft.Var[(1, 10), "float32>=0", "input"]
        d_b: ft.Var[(10,), "float32", "inout"]
        d_c: ft.Var[(10,), "float32", "inout"]
        for i in range(9, -1, -1):
            # HERE WE STILL NEED A TAPE
            d_b[i] += 2 * d_c[i] * b_tape[0, i]
            d_c[i] = 0
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
                                              ft.GradTapeMode.All,
                                              reset_provided_grad=False)
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
        ("d_y", (4,), "float32", "input", "cpu"),
    ]) as (d_x1, d_x2, d_x3, d_y):
        with ft.VarDef([("t.tape", (4,), "float32", "input", "cpu"),
                        ("d_t", (4,), "float32", "cache", "cpu"),
                        ("u.tape", (4,), "float32", "input", "cpu")
                       ]) as (t, d_t, u):
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
                                              ft.GradTapeMode.Nothing,
                                              reset_provided_grad=False)
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
                    ("d_y", (4,), "float32", "input", "cpu")
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
                                              ft.GradTapeMode.NoReuseOnly,
                                              reset_provided_grad=False)
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
                    ("d_y", (4,), "float32", "input", "cpu")
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


def test_inout_tape():

    @ft.transform
    def test(x: ft.Var[(), "float32", "inout"]):
        x[...] = ft.square(x[...])
        x[...] = ft.square(x[...])

    fwd, bwd, _, _ = ft.grad(test, ["x"], ["x"], ft.GradTapeMode.All, verbose=1)
    fwd = ft.lower(fwd, verbose=1)
    bwd = ft.lower(bwd, verbose=1)

    @ft.transform
    def fwd_std(x_tape: ft.Var[(2,), "float32", "output"],
                x: ft.Var[(), "float32", "inout"]):
        x_tape[0] = x[...]
        x[...] = ft.square(x[...])
        x_tape[1] = x[...]
        x[...] = ft.square(x[...])

    assert fwd_std.body.match(fwd.body)

    @ft.transform
    def bwd_std(x_tape: ft.Var[(2,), "float32", "input"],
                dx: ft.Var[(), "float32", "inout"]):
        dx[...] *= 4 * x_tape[1] * x_tape[0]

    assert bwd_std.body.match(bwd.body)


def test_inout_tape_written_only_once():

    @ft.transform
    def test(x: ft.Var[(), "float32", "inout"]):
        x[...] = ft.square(x[...])

    fwd, bwd, _, _ = ft.grad(test, ["x"], ["x"], ft.GradTapeMode.All, verbose=1)
    fwd = ft.lower(fwd, verbose=1)
    bwd = ft.lower(bwd, verbose=1)

    @ft.transform
    def fwd_std(x_tape: ft.Var[(1,), "float32", "output"],
                x: ft.Var[(), "float32", "inout"]):
        x_tape[0] = x[...]
        x[...] = ft.square(x[...])

    assert fwd_std.body.match(fwd.body)

    @ft.transform
    def bwd_std(x_tape: ft.Var[(1,), "float32", "input"],
                dx: ft.Var[(), "float32", "inout"]):
        dx[...] *= 2 * x_tape[0]

    assert bwd_std.body.match(bwd.body)


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
