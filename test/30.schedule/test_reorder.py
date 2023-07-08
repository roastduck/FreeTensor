import freetensor as ft
import pytest


def test_basic():
    with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                y[i, j] = i + j
    ast = ft.pop_ast(verbose=True)
    ast = ft.schedule(ast, lambda s: s.reorder(["L2", "L1"]), verbose=1)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ft.For("j", 0, 8) as j:
            with ft.For("i", 0, 4) as i:
                y[i, j] = i + j
    std = ft.pop_ast()

    assert std.match(ast)


def test_multiple_loops():
    with ft.VarDef("y", (4, 8, 16), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                with ft.For("k", 0, 16, label="L3") as k:
                    y[i, j, k] = (i + j) * k
    ast = ft.pop_ast(verbose=True)
    ast = ft.schedule(ast, lambda s: s.reorder(["L3", "L2", "L1"]), verbose=1)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("y", (4, 8, 16), "int32", "output", "cpu") as y:
        with ft.For("k", 0, 16) as k:
            with ft.For("j", 0, 8) as j:
                with ft.For("i", 0, 4) as i:
                    y[i, j, k] = (i + j) * k
    std = ft.pop_ast()

    assert std.match(ast)


def test_if_in_between():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.If(x[i] > 0):
                with ft.For("j", 0, 8, label="L2") as j:
                    y[i, j] = i + j
    ast = ft.pop_ast(verbose=True)
    ast = ft.schedule(ast, lambda s: s.reorder(["L2", "L1"]), verbose=1)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        with ft.For("j", 0, 8) as j:
            with ft.For("i", 0, 4) as i:
                with ft.If(x[i] > 0):
                    y[i, j] = i + j
    std = ft.pop_ast()

    assert std.match(ast)


def test_move_out_imperfect_stmt_in_between_before():
    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4,), "int32", "output", "cpu")]) as (y, z):
        with ft.For("i", 0, 4, label="L1") as i:
            z[i] = i
            with ft.For("j", 0, 8, label="L2") as j:
                y[i, j] = i + j
    ast = ft.pop_ast(verbose=True)
    ast = ft.schedule(
        ast,
        lambda s: s.reorder(["L2", "L1"], ft.ReorderMode.MoveOutImperfect),
        verbose=1)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4,), "int32", "output", "cpu")]) as (y, z):
        with ft.For("i", 0, 4, label="L1") as i:
            z[i] = i
        with ft.For("j", 0, 8, label="L2") as j:
            with ft.For("i", 0, 4, label="L1") as i:
                y[i, j] = i + j
    std = ft.pop_ast()

    assert std.match(ast)


def test_move_out_imperfect_stmt_in_between_after():
    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4,), "int32", "output", "cpu")]) as (y, z):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                y[i, j] = i + j
            z[i] = i
    ast = ft.pop_ast(verbose=True)
    ast = ft.schedule(
        ast,
        lambda s: s.reorder(["L2", "L1"], ft.ReorderMode.MoveOutImperfect),
        verbose=1)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4,), "int32", "output", "cpu")]) as (y, z):
        with ft.For("j", 0, 8, label="L2") as j:
            with ft.For("i", 0, 4, label="L1") as i:
                y[i, j] = i + j
        with ft.For("i", 0, 4, label="L1") as i:
            z[i] = i
    std = ft.pop_ast()

    assert std.match(ast)


def test_move_in_imperfect_stmt_in_between_before():
    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4,), "int32", "output", "cpu")]) as (y, z):
        with ft.For("i", 0, 4, label="L1") as i:
            z[i] = i
            with ft.For("j", 0, 8, label="L2") as j:
                y[i, j] = i + j
    ast = ft.pop_ast(verbose=True)
    ast = ft.schedule(
        ast,
        lambda s: s.reorder(["L2", "L1"], ft.ReorderMode.MoveInImperfect),
        verbose=1)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4,), "int32", "output", "cpu")]) as (y, z):
        with ft.For("j", 0, 8) as j:
            with ft.For("i", 0, 4) as i:
                with ft.If(j == 0):
                    z[i] = i
                y[i, j] = i + j
    std = ft.pop_ast()

    assert std.match(ast)


def test_move_in_imperfect_stmt_in_between_after():
    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4,), "int32", "output", "cpu")]) as (y, z):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                y[i, j] = i + j
            z[i] = i
    ast = ft.pop_ast(verbose=True)
    ast = ft.schedule(
        ast,
        lambda s: s.reorder(["L2", "L1"], ft.ReorderMode.MoveInImperfect),
        verbose=1)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4,), "int32", "output", "cpu")]) as (y, z):
        with ft.For("j", 0, 8) as j:
            with ft.For("i", 0, 4) as i:
                y[i, j] = i + j
                with ft.If(j == 7):
                    z[i] = i
    std = ft.pop_ast()

    assert std.match(ast)


def test_move_in_imperfect_loop_in_between():
    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4, 8), "int32", "output", "cpu")]) as (y, z):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                z[i, j] = i * j
            with ft.For("j", 0, 8, label="L3") as j:
                y[i, j] = i + j
    ast = ft.pop_ast(verbose=True)
    ast = ft.schedule(
        ast,
        lambda s: s.reorder(["L3", "L1"], ft.ReorderMode.MoveInImperfect),
        verbose=1)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4, 8), "int32", "output", "cpu")]) as (y, z):
        with ft.For("j", 0, 8) as j:
            with ft.For("i", 0, 4) as i:
                with ft.If(j == 0):
                    with ft.For("j1", 0, 8) as j1:
                        z[i, j1] = i * j1
                y[i, j] = i + j
    std = ft.pop_ast()

    assert std.match(ast)


def test_move_in_imperfect_multiple_loops_in_between_separated_by_vardef():
    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4, 8), "int32", "output", "cpu"),
                    ("w", (4, 8), "int32", "output", "cpu"),
                    ("n", (), "int32", "input", "cpu")]) as (y, z, w, n):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                z[i, j] = i * j
            with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                t[...] = n[...] * i
                with ft.For("j", 0, 8, label="L2") as j:
                    w[i, j] = t[...]
            with ft.For("j", 0, 8, label="L3") as j:
                y[i, j] = i + j
    ast = ft.pop_ast(verbose=True)
    ast = ft.schedule(
        ast,
        lambda s: s.reorder(["L3", "L1"], ft.ReorderMode.MoveInImperfect),
        verbose=1)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (4, 8), "int32", "output", "cpu"),
                    ("w", (4, 8), "int32", "output", "cpu"),
                    ("n", (), "int32", "input", "cpu")]) as (y, z, w, n):
        with ft.For("j", 0, 8) as j:
            with ft.For("i", 0, 4) as i:
                with ft.If(j == 0):
                    with ft.For("j1", 0, 8) as j1:
                        z[i, j1] = i * j1
                    with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                        t[...] = n[...] * i
                        with ft.For("j2", 0, 8) as j2:
                            w[i, j2] = t[...]
                y[i, j] = i + j
    std = ft.pop_ast()

    assert std.match(ast)


def test_legal_dependence():
    with ft.VarDef("y", (8,), "int32", "inout", "cpu") as y:
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                y[j] = (y[j] + 1) * j
    ast = ft.pop_ast(verbose=True)
    ast = ft.schedule(ast, lambda s: s.reorder(["L2", "L1"]), verbose=1)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("y", (8,), "int32", "inout", "cpu") as y:
        with ft.For("j", 0, 8) as j:
            with ft.For("i", 0, 4) as i:
                y[j] = (y[j] + 1) * j
    std = ft.pop_ast()

    assert std.match(ast)


def test_legal_dependence_only_inner_loops():
    with ft.VarDef("y", (16,), "int32", "inout", "cpu") as y:
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                with ft.For("k", 0, 16, label="L3") as k:
                    y[k] = (y[k] + 1) * j * k
    ast = ft.pop_ast(verbose=True)
    ast = ft.schedule(ast, lambda s: s.reorder(["L3", "L2"]), verbose=1)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("y", (16,), "int32", "inout", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            with ft.For("k", 0, 16) as k:
                with ft.For("j", 0, 8) as j:
                    y[k] = (y[k] + 1) * j * k
    std = ft.pop_ast()

    assert std.match(ast)


def test_illegal_dependence():
    with ft.VarDef("y", (1,), "int32", "output", "cpu") as y:
        y[0] = 0
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                y[0] = y[0] * i + j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.reorder(["L2", "L1"])
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_move_in_imperfect_illegal_dependence_of_stmt_in_between():
    with ft.VarDef([("y", (4, 8), "int32", "output", "cpu"),
                    ("z", (), "int32", "cache", "cpu")]) as (y, z):
        with ft.For("i", 0, 4, label="L1") as i:
            z[()] = i * i
            with ft.For("j", 0, 8, label="L2") as j:
                y[i, j] = z[()] + j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.reorder(["L2", "L1"], ft.ReorderMode.MoveInImperfect)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_move_in_imperfect_illegal_dependence_of_stmt_in_between_on_local_var():

    @ft.transform
    def f(x: ft.Var[(4,), "int32", "input"], y: ft.Var[(4,), "int32", "inout"]):
        #! label: L1
        for i in range(0, 4):
            t = ft.empty((), "int32")
            t[...] = x[i] * 2
            #! label: L2
            for j in range(0, 4):
                y[i] += t[...] * j

    s = ft.Schedule(f, verbose=2)
    with pytest.raises(ft.InvalidSchedule):
        s.reorder(["L2", "L1"], ft.ReorderMode.MoveInImperfect)


def test_reduction():
    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (1,), "int32", "output", "cpu")]) as (x, y):
        y[0] = 0
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                y[0] = y[0] + x[i, j]
    ast = ft.pop_ast(verbose=True)
    ast = ft.schedule(ast, lambda s: s.reorder(["L2", "L1"]), verbose=1)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (1,), "int32", "output", "cpu")]) as (x, y):
        y[0] = 0
        with ft.For("j", 0, 8) as j:
            with ft.For("i", 0, 4) as i:
                ft.Any()
    std = ft.pop_ast()

    assert std.match(ast)


def test_local_var():
    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y1", (4, 8), "int32", "output", "cpu"),
        ("y2", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y1, y2):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                with ft.VarDef("buf", (1,), "int32", "cache", "cpu") as buf:
                    buf[0] = x0[i, j] + x1[i, j]
                    y1[i, j] = buf[0] * 2
                    y2[i, j] = buf[0] * 3
    ast = ft.pop_ast(verbose=True)
    ast = ft.schedule(ast, lambda s: s.reorder(["L2", "L1"]), verbose=1)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y1", (4, 8), "int32", "output", "cpu"),
        ("y2", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y1, y2):
        with ft.For("j", 0, 8) as j:
            with ft.For("i", 0, 4) as i:
                with ft.VarDef("buf", (1,), "int32", "cache", "cpu") as buf:
                    buf[0] = x0[i, j] + x1[i, j]
                    y1[i, j] = buf[0] * 2
                    y2[i, j] = buf[0] * 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_dep_by_external_var():
    with ft.VarDef([("idx1", (4, 8), "int32", "input", "cpu"),
                    ("idx2", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 9), "int32", "output", "cpu")]) as (idx1, idx2,
                                                                  y):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                y[idx1[i, j] + i, idx2[i, j] + j] = i + j
                # May be overriden by some next statements, for example,
                # when (i, j) == (0, 1), idx1[i, j] + i == 0 + 0 == 0, and
                # when (i, j) == (1, 0), idx1[i, j] + i == -1 + 1 == 0, and so is to j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=2)
    with pytest.raises(ft.InvalidSchedule):
        s.reorder(["L2", "L1"])


def test_no_dep_by_invariant_external_var():
    with ft.VarDef([("idx1", (), "int32", "input", "cpu"),
                    ("idx2", (), "int32", "input", "cpu"),
                    ("y", (4, 9), "int32", "output", "cpu")]) as (idx1, idx2,
                                                                  y):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, 8, label="L2") as j:
                y[idx1[...] + i, idx2[...] + j] = i + j
                # Nothing will be overwridden, because idx1 and idx2 always keep the same.
                # After adding with i or j, they are all different among iterations
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=2)
    s.reorder(["L2", "L1"])
    ast = s.ast()
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("idx1", (), "int32", "input", "cpu"),
                    ("idx2", (), "int32", "input", "cpu"),
                    ("y", (4, 9), "int32", "output", "cpu")]) as (idx1, idx2,
                                                                  y):
        with ft.For("j", 0, 8, label="L2") as j:
            with ft.For("i", 0, 4, label="L1") as i:
                y[idx1[...] + i, idx2[...] + j] = i + j
    std = ft.pop_ast()

    assert std.match(ast)


def test_no_dep_by_external_var_variant_of_another_loop():
    with ft.VarDef([("idx1", (4,), "int32", "input", "cpu"),
                    ("idx2", (4,), "int32", "input", "cpu"),
                    ("y", (4, 9), "int32", "output", "cpu")]) as (idx1, idx2,
                                                                  y):
        with ft.For("k", 0, 4, label="L0") as k:
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", 0, 8, label="L2") as j:
                    y[idx1[k] + i, idx2[k] + j] = i + j
                    # Nothing will be overwridden. Although idx1 and idx2 are related to k,
                    # we only reorder with respect to the same k, so they can be treated
                    # as constant values. After adding with i or j, they are all different
                    # among iterations
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=2)
    s.reorder(["L2", "L1"])
    ast = s.ast()
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("idx1", (4,), "int32", "input", "cpu"),
                    ("idx2", (4,), "int32", "input", "cpu"),
                    ("y", (4, 9), "int32", "output", "cpu")]) as (idx1, idx2,
                                                                  y):
        with ft.For("k", 0, 4, label="L0") as k:
            with ft.For("j", 0, 8, label="L2") as j:
                with ft.For("i", 0, 4, label="L1") as i:
                    y[idx1[k] + i, idx2[k] + j] = i + j
    std = ft.pop_ast()

    assert std.match(ast)


def test_dep_by_external_var_reversed_loop():
    with ft.VarDef([("idx1", (4, 8), "int32", "input", "cpu"),
                    ("idx2", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 9), "int32", "output", "cpu")]) as (idx1, idx2,
                                                                  y):
        with ft.For("i", 3, -1, -1, label="L1") as i:
            with ft.For("j", 7, -1, -1, label="L2") as j:
                y[idx1[i, j] + i, idx2[i, j] + j] = i + j
                # May be overriden by some next statements, for example,
                # when (i, j) == (0, 1), idx1[i, j] + i == 0 + 0 == 0, and
                # when (i, j) == (1, 0), idx1[i, j] + i == -1 + 1 == 0, and so is to j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=2)
    with pytest.raises(ft.InvalidSchedule):
        s.reorder(["L2", "L1"])


def test_no_merge_if_outer_iter_var_is_used_in_inner():
    with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", 0, i, label="L2") as j:
                y[i, j] = i * j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=2)
    with pytest.raises(ft.InvalidSchedule):
        s.reorder(["L2", "L1"])
