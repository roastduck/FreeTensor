import freetensor as ft
import pytest


def test_basic():
    with ft.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ft.For("i", 0, 4, nid="L1") as i:
            ft.MarkNid("S1")
            y1[i] = i + 1
            ft.MarkNid("S2")
            y2[i] = i + 2
            ft.MarkNid("S3")
            y3[i] = i + 3
            ft.MarkNid("S4")
            y4[i] = i + 4
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.swap(["S2", "S3", "S1"])
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ft.For("i", 0, 4) as i:
            y2[i] = i + 2
            y3[i] = i + 3
            y1[i] = i + 1
            y4[i] = i + 4
    std = ft.pop_ast()

    assert std.match(ast)


def test_not_consecutive():
    with ft.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ft.For("i", 0, 4, nid="L1") as i:
            ft.MarkNid("S1")
            y1[i] = i + 1
            ft.MarkNid("S2")
            y2[i] = i + 2
            ft.MarkNid("S3")
            y3[i] = i + 3
            ft.MarkNid("S4")
            y4[i] = i + 4
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.swap(["S4", "S1"])
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_dependency():
    with ft.VarDef([("y1", (4,), "int32", "inout", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4, nid="L1") as i:
            ft.MarkNid("S1")
            y1[i] = i + 1
            ft.MarkNid("S2")
            y2[i] = y1[i] * 2
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.swap(["S2", "S1"])
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)
