import freetensor as ft
import pytest


def test_basic():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ft.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_multiple_assignments():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 2) as i:
                t[i] = x[i] * 2
            with ft.For("i", 2, 4) as i:
                t[i] = x[i] * 3
            with ft.For("i", 0, 2) as i:
                y[i] = t[i] + 1
            with ft.For("i", 2, 4) as i:
                y[i] = t[i] + 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 2) as i:
            y[i] = x[i] * 2 + 1
        with ft.For("i", 2, 4) as i:
            y[i] = x[i] * 3 + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_modified_unrelated_item():
    with ft.VarDef([("x", (5,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            x[4] = 10
            with ft.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (5,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        x[4] = 10
        with ft.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_loop_around():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 3), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ft.For("j", 0, 4) as j:
                with ft.If(j > 0):
                    with ft.For("i", 0, 4) as i:
                        y[j - 1, i] = t[i] + 1
                with ft.For("i", 0, 4) as i:
                    t[i] = x[i] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 3), "int32", "output", "cpu")]) as (x, y):
        with ft.For("j", 1, 4) as j:
            with ft.For("i", 0, 4) as i:
                y[j + -1, i] = x[i] * 2 + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_loop_around_different_iter_no_prop():
    with ft.VarDef([("x", (4, 3), "int32", "input", "cpu"),
                    ("y", (4, 3), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ft.For("j", 0, 4) as j:
                with ft.If(j > 0):
                    with ft.For("i", 0, 4) as i:
                        y[j + -1, i] = t[i] + 1
                with ft.For("i", 0, 4) as i:
                    t[i] = x[j, i] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.inline("T")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_no_inline_expr_is_changed_1():
    with ft.VarDef([("x", (4,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ft.For("i", 0, 4) as i:
                x[i] = 1
            with ft.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.inline("T")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_no_inline_expr_is_changed_2():
    with ft.VarDef([("x", (4,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = x[i] * 2
                x[i] = 1
            with ft.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.inline("T")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_no_inline_expr_is_changed_multiple_times():
    with ft.VarDef([("x", (4,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ft.For("i", 0, 4) as i:
                x[i] = i + 1
            with ft.For("i", 0, 4) as i:
                y[i] = x[i]
            with ft.For("i", 0, 4) as i:
                x[i] = i + 2
            with ft.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.inline("T")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_no_inline_output_var():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "output", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ft.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.inline("T")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_different_iter_with_the_same_name():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ft.For("i", 4, 8) as i:
                y[i + -4] = t[i + -4] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 4, 8) as i:
            y[i + -4] = x[i + -4] * 2 + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_different_iter_with_different_names():
    with ft.VarDef([("x", (4,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ft.For("j", 4, 8) as j:
                y[j + -4] = t[j + -4] + 1
            with ft.For("i", 0, 4) as i:
                x[i] = 0
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("j", 4, 8) as j:
            y[j + -4] = x[j + -4] * 2 + 1
        with ft.For("i", 0, 4) as i:
            x[i] = 0
    std = ft.pop_ast()

    assert std.match(ast)


def test_different_iter_with_uncertain_offset_no_inline():
    with ft.VarDef([("offset", (), "int32", "inout", "cpu"),
                    ("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (offset, x, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ft.For("i", offset[()], offset[()] + 4) as i:
                t[i + -1 * offset[()]] = x[i + -1 * offset[()]] * 2
            offset[()] = 0
            with ft.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.inline("T")
    ast_ = s.ast()  # Should not changed
    print(ast_)
    assert ast_.match(ft.make_reduction(ast))


def test_different_iter_non_linear():
    with ft.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("y", (16,), "int32", "output", "cpu")]) as (x1, x2, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (16,), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                with ft.For("j", 0, 4) as j:
                    t[i * 4 + j] = x1[i] * x2[j]
            with ft.For("k", 0, 16) as k:
                y[k] = t[k] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=["use_builtin_div"], verbose=1)

    with ft.VarDef([("x1", (4,), "int32", "input", "cpu"),
                    ("x2", (4,), "int32", "input", "cpu"),
                    ("y", (16,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ft.For("k", 0, 16) as k:
            y[k] = x1[k // 4] * x2[k % 4] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_inline_serial_into_parallel():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ft.For("i", 0, 4, nid="L") as i:
                y[i] = t[i] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.parallelize("L", "threadIdx.x")
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_correct_scope():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("U")
        with ft.VarDef("u", (4,), "int32", "cache", "cpu") as u:
            ft.MarkNid("T")
            with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
                with ft.For("i", 0, 4) as i:
                    t[i] = x[i] * 2
                with ft.For("i", 0, 4) as i:
                    u[i] = t[i] + 1
            with ft.For("i", 0, 4) as i:
                y[i] = u[i] - 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.inline("U")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=['prop_one_time_use'], verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("T")
        with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ft.For("i", 0, 4) as i:
                y[i] = t[i]
    std = ft.pop_ast()

    assert std.match(ast)


def test_no_inline_defined_inside_a_loop():
    with ft.VarDef([("x", (4, 4), "int32", "input", "cpu"),
                    ("y", (4, 4), "int32", "output", "cpu")]) as (x, y):
        ft.MarkNid("U")
        with ft.VarDef("u", (4, 4), "int32", "cache", "cpu") as u:
            with ft.For("i", 0, 4) as i:
                ft.MarkNid("T")
                with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
                    with ft.For("j", 0, 4) as j:
                        t[j] = x[i, j] * 2
                    with ft.For("j", 0, 4) as j:
                        u[i, j] = t[j] + 1
            with ft.For("i", 0, 4) as i:
                with ft.For("j", 0, 4) as j:
                    y[i, j] = u[i, j] + -1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.inline("U")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ft.make_reduction(ast))


def test_def_then_use_then_def_in_the_same_place():
    with ft.VarDef("x", (128, 128, 128), "float64", "inout", "cpu") as x:
        with ft.VarDef("r", (128, 128, 128), "float64", "input", "cpu") as r:
            with ft.For("ix_2", 1, 127) as ix:
                with ft.For("iy_2", 1, 127) as iy:
                    with ft.For("iz_2", 1, 127) as iz:
                        ft.MarkNid('sx')
                        with ft.VarDef("sx", (), "float64", "cache",
                                       "cpu") as sx:
                            sx[()] = x[ix + -1, iy + 1, iz + -1] + x[ix, iy + 1,
                                                                     iz + -1]
                            x[ix, iy, iz] = sx[()] / 26.0
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.inline('sx')
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("x", (128, 128, 128), "float64", "inout", "cpu") as x:
        with ft.VarDef("r", (128, 128, 128), "float64", "input", "cpu") as r:
            with ft.For("ix_2", 1, 127) as ix:
                with ft.For("iy_2", 1, 127) as iy:
                    with ft.For("iz_2", 1, 127) as iz:
                        x[ix, iy, iz] = (x[ix + -1, iy + 1, iz + -1] +
                                         x[ix, iy + 1, iz + -1]) / 26.0
    std = ft.pop_ast()

    assert std.match(ast)
