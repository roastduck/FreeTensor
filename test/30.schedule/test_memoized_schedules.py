import freetensor as ft
import pytest


def test_memoize_success():
    with ft.VarDef("y", (8,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8, nid="L1") as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    s1 = ft.Schedule(ast)
    s2 = s1.fork()
    s1.split("L1", 4)
    ast1 = s1.ast()
    s2.split("L1", 4)
    ast2 = s2.ast()

    assert ast1 is ast2  # Same object, memoized


def test_memoize_failure():
    with ft.VarDef("y", (1,), "int32", "output", "cpu") as y:
        y[0] = 0
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                y[0] = y[0] * i + j
    ast = ft.pop_ast(verbose=True)
    s1 = ft.Schedule(ast)
    s2 = s1.fork()
    with pytest.raises(ft.InvalidSchedule):
        s1.reorder(["L2", "L1"])
    with pytest.raises(ft.InvalidSchedule):
        s2.reorder(["L2", "L1"])
    assert s1.ast().match(ast)  # Should not changed
    assert s2.ast().match(ast)  # Should not changed
