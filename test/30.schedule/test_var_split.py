import freetensor as ft
import pytest


def test_factor():
    with ft.VarDef("z", (), "int32", "inout", "cpu") as z:
        ft.MarkLabel("Dy")
        with ft.VarDef("y", (8,), "int32", "cache", "cpu") as y:
            with ft.For("i", 0, 8) as i:
                y[i] = i
            with ft.For("i", 0, 8) as i:
                z[...] += y[i]
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.var_split("Dy", 0, ft.VarSplitMode.FixedSize, 4)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast,
                   skip_passes=[
                       "use_builtin_div", "tensor_prop_const",
                       "prop_one_time_use"
                   ],
                   verbose=1)

    with ft.VarDef("z", (), "int32", "inout", "cpu") as z:
        with ft.VarDef("y", (2, 4), "int32", "cache", "cpu") as y:
            with ft.For("i", 0, 8) as i:
                y[i // 4, i % 4] = i
            with ft.For("i", 0, 8) as i:
                z[...] += y[i // 4, i % 4]
    std = ft.pop_ast()

    assert std.match(ast)


def test_nparts():
    with ft.VarDef("z", (), "int32", "inout", "cpu") as z:
        ft.MarkLabel("Dy")
        with ft.VarDef("y", (8,), "int32", "cache", "cpu") as y:
            with ft.For("i", 0, 8) as i:
                y[i] = i
            with ft.For("i", 0, 8) as i:
                z[...] += y[i]
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.var_split("Dy", 0, ft.VarSplitMode.FixedSize, nparts=4)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast,
                   skip_passes=[
                       "use_builtin_div", "tensor_prop_const",
                       "prop_one_time_use"
                   ],
                   verbose=1)

    with ft.VarDef("z", (), "int32", "inout", "cpu") as z:
        with ft.VarDef("y", (4, 2), "int32", "cache", "cpu") as y:
            with ft.For("i", 0, 8) as i:
                y[i // 2, i % 2] = i
            with ft.For("i", 0, 8) as i:
                z[...] += y[i // 2, i % 2]
    std = ft.pop_ast()

    assert std.match(ast)


def test_non_divisible():
    with ft.VarDef("z", (), "int32", "inout", "cpu") as z:
        ft.MarkLabel("Dy")
        with ft.VarDef("y", (10,), "int32", "cache", "cpu") as y:
            with ft.For("i", 0, 10) as i:
                y[i] = i
            with ft.For("i", 0, 10) as i:
                z[...] += y[i]
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.var_split("Dy", 0, ft.VarSplitMode.FixedSize, 4)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast,
                   skip_passes=[
                       "use_builtin_div", "tensor_prop_const",
                       "prop_one_time_use"
                   ],
                   verbose=1)

    with ft.VarDef("z", (), "int32", "inout", "cpu") as z:
        with ft.VarDef("y", (3, 4), "int32", "cache", "cpu") as y:
            with ft.For("i", 0, 10) as i:
                y[i // 4, i % 4] = i
            with ft.For("i", 0, 10) as i:
                z[...] += y[i // 4, i % 4]
    std = ft.pop_ast()

    assert std.match(ast)


def test_view_of_io_var():
    ft.MarkLabel("Dy")
    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8) as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.var_split("Dy", 0, ft.VarSplitMode.FixedSize, 4)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=["use_builtin_div"], verbose=1)

    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8) as i:
            with ft.VarDef("y.view", (2, 4),
                           "int32",
                           "cache",
                           "cpu",
                           view_of="y") as y_view:
                y_view[i // 4, i % 4] = i
    std = ft.pop_ast()

    assert std.match(ast)


def test_guard_view_when_caching():
    ft.MarkLabel("Dx")
    with ft.VarDef("x", (10,), "int32", "input", "cpu") as x:
        with ft.VarDef([("y", (10,), "int32", "output", "cpu"),
                        ("z", (10,), "int32", "output", "cpu")]) as (y, z):
            with ft.For("i", 0, 10, label="Li") as i:
                y[i] = x[i] + 1
                z[i] = x[i] + 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.var_split("Dx", 0, ft.VarSplitMode.FixedSize, 4)
    s.cache("Li", "x.view", "cpu")
    ast = s.ast()
    print(ast)

    with ft.VarDef([("x", (10,), "int32", "input", "cpu"),
                    ("y", (10,), "int32", "output", "cpu"),
                    ("z", (10,), "int32", "output", "cpu")]) as (x, y, z):
        with ft.VarDef("t", (3, 4), "int32", "cache", "cpu") as t:
            with ft.For("i", 0, 3) as i:
                with ft.For("j", 0, 4) as j:
                    with ft.If(i * 4 + j < 10):
                        with ft.VarDef("x.view", (3, 4),
                                       "int32",
                                       "cache",
                                       "cpu",
                                       view_of="x") as x_view:
                            t[i, j] = x_view[i, j]
            with ft.For("k", 0, 10, label="Li") as k:
                y[k] = t[k // 4, k % 4] + 1
                z[k] = t[k // 4, k % 4] + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_not_found():
    ft.MarkLabel("Dy")
    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8) as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.var_split("Dx", 0, ft.VarSplitMode.FixedSize)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_out_of_range():
    ft.MarkLabel("Dy")
    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8) as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.var_split("Dy", 1, ft.VarSplitMode.FixedSize)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)
