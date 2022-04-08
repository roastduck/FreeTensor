import ir
import pytest


def test_basic():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ir.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_multiple_assignments():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 2) as i:
                t[i] = x[i] * 2
            with ir.For("i", 2, 4) as i:
                t[i] = x[i] * 3
            with ir.For("i", 0, 2) as i:
                y[i] = t[i] + 1
            with ir.For("i", 2, 4) as i:
                y[i] = t[i] + 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 2) as i:
            y[i] = x[i] * 2 + 1
        with ir.For("i", 2, 4) as i:
            y[i] = x[i] * 3 + 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_modified_unrelated_item():
    with ir.VarDef([("x", (5,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            x[4] = 10
            with ir.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (5,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        x[4] = 10
        with ir.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_loop_around():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 3), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("j", 0, 4) as j:
                with ir.If(j > 0):
                    with ir.For("i", 0, 4) as i:
                        y[j - 1, i] = t[i] + 1
                with ir.For("i", 0, 4) as i:
                    t[i] = x[i] * 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 3), "int32", "output", "cpu")]) as (x, y):
        with ir.For("j", 1, 4) as j:
            with ir.For("i", 0, 4) as i:
                y[j + -1, i] = x[i] * 2 + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_loop_around_different_iter_no_prop():
    with ir.VarDef([("x", (4, 3), "int32", "input", "cpu"),
                    ("y", (4, 3), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("j", 0, 4) as j:
                with ir.If(j > 0):
                    with ir.For("i", 0, 4) as i:
                        y[j + -1, i] = t[i] + 1
                with ir.For("i", 0, 4) as i:
                    t[i] = x[j, i] * 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.inline("T")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_no_inline_expr_is_changed_1():
    with ir.VarDef([("x", (4,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ir.For("i", 0, 4) as i:
                x[i] = 1
            with ir.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.inline("T")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_no_inline_expr_is_changed_2():
    with ir.VarDef([("x", (4,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x[i] * 2
                x[i] = 1
            with ir.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.inline("T")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_no_inline_expr_is_changed_multiple_times():
    with ir.VarDef([("x", (4,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ir.For("i", 0, 4) as i:
                x[i] = i + 1
            with ir.For("i", 0, 4) as i:
                y[i] = x[i]
            with ir.For("i", 0, 4) as i:
                x[i] = i + 2
            with ir.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.inline("T")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_no_inline_output_var():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "output", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ir.For("i", 0, 4) as i:
                y[i] = t[i] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.inline("T")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_different_iter_with_the_same_name():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ir.For("i", 4, 8) as i:
                y[i + -4] = t[i + -4] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 4, 8) as i:
            y[i + -4] = x[i + -4] * 2 + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_different_iter_with_different_names():
    with ir.VarDef([("x", (4,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ir.For("j", 4, 8) as j:
                y[j + -4] = t[j + -4] + 1
            with ir.For("i", 0, 4) as i:
                x[i] = 0
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("j", 4, 8) as j:
            y[j + -4] = x[j + -4] * 2 + 1
        with ir.For("i", 0, 4) as i:
            x[i] = 0
    std = ir.pop_ast()

    assert std.match(ast)


#def test_different_iter_non_affine():
#    with ir.VarDef([("x1", (4,), "int32", "input", "cpu"),
#                    ("x2", (4,), "int32", "input", "cpu"),
#                    ("y", (16,), "int32", "output", "cpu")]) as (x1, x2, y):
#        ir.MarkNid("T")
#        with ir.VarDef("t", (16,), "int32", "cache", "cpu") as t:
#            with ir.For("i", 0, 4) as i:
#                with ir.For("j", 0, 4) as j:
#                    t[i * 4 + j] = x1[i] * x2[j]
#            with ir.For("k", 0, 16) as k:
#                y[k] = t[k] + 1
#    ast = ir.pop_ast()
#    print(ast)
#    s = ir.Schedule(ast)
#    s.inline("T")
#    ast = s.ast()
#    print(ast)
#    ast = ir.lower(ast)
#    print(ast)
#
#    assert False


def test_inline_serial_into_parallel():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ir.For("i", 0, 4, nid="L") as i:
                y[i] = t[i] + 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.parallelize("L", "threadIdx.x")
    s.inline("T")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = x[i] * 2 + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_correct_scope():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("U")
        with ir.VarDef("u", (4,), "int32", "cache", "cpu") as u:
            ir.MarkNid("T")
            with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
                with ir.For("i", 0, 4) as i:
                    t[i] = x[i] * 2
                with ir.For("i", 0, 4) as i:
                    u[i] = t[i] + 1
            with ir.For("i", 0, 4) as i:
                y[i] = u[i] - 1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.inline("U")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("T")
        with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
            with ir.For("i", 0, 4) as i:
                t[i] = x[i] * 2
            with ir.For("i", 0, 4) as i:
                y[i] = t[i]
    std = ir.pop_ast()

    assert std.match(ast)


def test_no_inline_defined_inside_a_loop():
    with ir.VarDef([("x", (4, 4), "int32", "input", "cpu"),
                    ("y", (4, 4), "int32", "output", "cpu")]) as (x, y):
        ir.MarkNid("U")
        with ir.VarDef("u", (4, 4), "int32", "cache", "cpu") as u:
            with ir.For("i", 0, 4) as i:
                ir.MarkNid("T")
                with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
                    with ir.For("j", 0, 4) as j:
                        t[j] = x[i, j] * 2
                    with ir.For("j", 0, 4) as j:
                        u[i, j] = t[j] + 1
            with ir.For("i", 0, 4) as i:
                with ir.For("j", 0, 4) as j:
                    y[i, j] = u[i, j] + -1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.inline("U")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ir.make_reduction(ast))
