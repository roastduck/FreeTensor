import freetensor as ft
import pytest


def test_factor():
    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8, label="L1") as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.split("L1", 4)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i.0", 0, 2) as i0:
            with ft.For("i.1", 0, 4) as i1:
                y[i1 + 4 * i0] = i1 + 4 * i0
    std = ft.pop_ast()

    assert std.match(ast)


def test_factor_with_step():
    with ft.VarDef("y", (16,), "int32", "output", "cpu") as y:
        with ft.For("i", 14, -2, -2, label="L1") as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.split("L1", 2)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("y", (16,), "int32", "output", "cpu") as y:
        with ft.For("i.0", 0, 4) as i0:
            with ft.For("i.1", 0, 2) as i1:
                y[14 + -2 * i1 + -4 * i0] = 14 + -2 * i1 + -4 * i0
    std = ft.pop_ast()

    assert std.match(ast)


def test_nparts():
    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8, label="L1") as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.split("L1", nparts=4)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i.0", 0, 4) as i0:
            with ft.For("i.1", 0, 2) as i1:
                y[i1 + 2 * i0] = i1 + 2 * i0
    std = ft.pop_ast()

    assert std.match(ast)


def test_nparts_with_step():
    with ft.VarDef("y", (16,), "int32", "output", "cpu") as y:
        with ft.For("i", 14, -2, -2, label="L1") as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.split("L1", nparts=2)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef("y", (16,), "int32", "output", "cpu") as y:
        with ft.For("i.0", 0, 2) as i0:
            with ft.For("i.1", 0, 4) as i1:
                y[14 + -2 * i1 + -8 * i0] = 14 + -2 * i1 + -8 * i0
    std = ft.pop_ast()

    assert std.match(ast)


def test_guard():
    with ft.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 10, label="L1") as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.split("L1", 4)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    # The result after simplifying the loop length
    with ft.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ft.For("i.0", 0, 3) as i0:
            with ft.For("i.1", 0, ft.any()) as i1:
                y[i1 + 4 * i0] = i1 + 4 * i0
    std = ft.pop_ast()

    assert std.match(ast)


def test_guard_with_step():
    with ft.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 10, 2, label="L1") as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.split("L1", 3)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    # The result after simplifying the loop length
    with ft.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ft.For("i.0", 0, 2) as i0:
            with ft.For("i.1", 0, ft.any()) as i1:
                y[2 * i1 + 6 * i0] = 2 * i1 + 6 * i0
    std = ft.pop_ast()

    assert std.match(ast)


def test_shift():
    with ft.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 10, label="L1") as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.split("L1", 4, -1, 1)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    # The result after simplifying the loop length
    with ft.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ft.For("i.0", 0, 3) as i0:
            with ft.For("i.1", ft.any(), ft.any()) as i1:
                y[i1 + 4 * i0 - 1] = i1 + 4 * i0 - 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_not_found():
    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8) as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.split("L1", 4)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_simplify_split_then_merge():
    with ft.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 10, label="L") as i:
            y[i] = i

    ast = ft.pop_ast()
    std = ast
    s = ft.Schedule(ast)
    L0, L1 = s.split("L", 4)
    L = s.merge(L0, L1)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    assert std.match(ast)
