import ir
import pytest


def test_factor():
    with ir.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 8, nid="L1") as i:
            y[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.split("L1", 4)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ir.For("i.0", 0, 2) as i0:
            with ir.For("i.1", 0, 4) as i1:
                y[i1 + 4 * i0] = i1 + 4 * i0
    std = ir.pop_ast()

    assert std.match(ast)


def test_factor_with_step():
    with ir.VarDef("y", (16,), "int32", "output", "cpu") as y:
        with ir.For("i", 14, -2, -2, nid="L1") as i:
            y[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.split("L1", 2)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (16,), "int32", "output", "cpu") as y:
        with ir.For("i.0", 0, 4) as i0:
            with ir.For("i.1", 0, 2) as i1:
                y[14 + -2 * i1 + -4 * i0] = 14 + -2 * i1 + -4 * i0
    std = ir.pop_ast()

    assert std.match(ast)


def test_nparts():
    with ir.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 8, nid="L1") as i:
            y[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.split("L1", nparts=4)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ir.For("i.0", 0, 4) as i0:
            with ir.For("i.1", 0, 2) as i1:
                y[i1 + 2 * i0] = i1 + 2 * i0
    std = ir.pop_ast()

    assert std.match(ast)


def test_nparts_with_step():
    with ir.VarDef("y", (16,), "int32", "output", "cpu") as y:
        with ir.For("i", 14, -2, -2, nid="L1") as i:
            y[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.split("L1", nparts=2)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (16,), "int32", "output", "cpu") as y:
        with ir.For("i.0", 0, 2) as i0:
            with ir.For("i.1", 0, 4) as i1:
                y[14 + -2 * i1 + -8 * i0] = 14 + -2 * i1 + -8 * i0
    std = ir.pop_ast()

    assert std.match(ast)


def test_guard():
    with ir.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 10, nid="L1") as i:
            y[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.split("L1", 4)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    # The result after simplifying the loop length
    with ir.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ir.For("i.0", 0, 3) as i0:
            with ir.For("i.1", 0, ir.any()) as i1:
                y[i1 + 4 * i0] = i1 + 4 * i0
    std = ir.pop_ast()

    assert std.match(ast)


def test_guard_with_step():
    with ir.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 10, 2, nid="L1") as i:
            y[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.split("L1", 3)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    # The result after simplifying the loop length
    with ir.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ir.For("i.0", 0, 2) as i0:
            with ir.For("i.1", 0, ir.any()) as i1:
                y[2 * i1 + 6 * i0] = 2 * i1 + 6 * i0
    std = ir.pop_ast()

    assert std.match(ast)


def test_not_found():
    with ir.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 8) as i:
            y[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.split("L1", 4)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_simplify_split_then_merge():
    with ir.VarDef("y", (10,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 10, nid="L") as i:
            y[i] = i

    ast = ir.pop_ast()
    std = ast
    s = ir.Schedule(ast)
    L0, L1 = s.split("L", 4)
    L = s.merge(L0, L1)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    assert std.match(ast)
