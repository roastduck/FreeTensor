import freetensor as ft
import pytest


def test_basic():
    with ft.VarDef("z", (), "int32", "inout", "cpu") as z:
        ft.MarkLabel("Dy")
        with ft.VarDef("y", (7, 8), "int32", "cache", "cpu") as y:
            with ft.For("i", 0, 7) as i:
                with ft.For("j", 0, 8) as j:
                    y[i, j] = i + j
            with ft.For("i", 0, 7) as i:
                with ft.For("j", 0, 8) as j:
                    z[...] += y[i, j]
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.var_merge("Dy", 0)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast,
                   skip_passes=[
                       "use_builtin_div", "tensor_prop_const",
                       "prop_one_time_use"
                   ],
                   verbose=1)

    with ft.VarDef("z", (), "int32", "inout", "cpu") as z:
        with ft.VarDef("y", (56,), "int32", "cache", "cpu") as y:
            with ft.For("i", 0, 7) as i:
                with ft.For("j", 0, 8) as j:
                    y[i * 8 + j] = i + j
            with ft.For("i", 0, 7) as i:
                with ft.For("j", 0, 8) as j:
                    z[...] += y[i * 8 + j]
    std = ft.pop_ast()

    assert std.match(ast)


def test_view_of_io_var():
    ft.MarkLabel("Dy")
    with ft.VarDef("y", (7, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 7) as i:
            with ft.For("j", 0, 8) as j:
                y[i, j] = i + j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.var_merge("Dy", 0)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=["use_builtin_div"], verbose=1)

    with ft.VarDef("y", (7, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 7) as i:
            with ft.For("j", 0, 8) as j:
                with ft.VarDef("y_view", (56,),
                               "int32",
                               "cache",
                               "cpu",
                               view_of="y") as y_view:
                    y_view[i * 8 + j] = i + j
    std = ft.pop_ast()

    assert std.match(ast)


def test_not_found():
    ft.MarkLabel("Dy")
    with ft.VarDef("y", (7, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 7) as i:
            with ft.For("j", 0, 8) as j:
                y[i, j] = i + j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.var_merge("Dx", 0)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_out_of_range():
    ft.MarkLabel("Dy")
    with ft.VarDef("y", (7, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 7) as i:
            with ft.For("j", 0, 8) as j:
                y[i, j] = i + j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.var_merge("Dy", 1)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)
