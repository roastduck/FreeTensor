import ir
import pytest


def test_basic():
    with ir.VarDef(
        [
            ("y1", (4,), "int32", "output", "cpu"),
            ("y2", (4,), "int32", "output", "cpu"),
            ("y3", (4,), "int32", "output", "cpu"),
            ("y4", (4,), "int32", "output", "cpu"),
        ]
    ) as (y1, y2, y3, y4):
        with ir.For("i", 0, 4, nid="L1") as i:
            ir.MarkNid("S1")
            y1[i] = i + 1
            ir.MarkNid("S2")
            y2[i] = i + 2
            ir.MarkNid("S3")
            y3[i] = i + 3
            ir.MarkNid("S4")
            y4[i] = i + 4
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.swap(["S2", "S3", "S1"])
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef(
        [
            ("y1", (4,), "int32", "output", "cpu"),
            ("y2", (4,), "int32", "output", "cpu"),
            ("y3", (4,), "int32", "output", "cpu"),
            ("y4", (4,), "int32", "output", "cpu"),
        ]
    ) as (y1, y2, y3, y4):
        with ir.For("i", 0, 4) as i:
            y2[i] = i + 2
            y3[i] = i + 3
            y1[i] = i + 1
            y4[i] = i + 4
    std = ir.pop_ast()

    assert std.match(ast)


def test_not_consecutive():
    with ir.VarDef(
        [
            ("y1", (4,), "int32", "output", "cpu"),
            ("y2", (4,), "int32", "output", "cpu"),
            ("y3", (4,), "int32", "output", "cpu"),
            ("y4", (4,), "int32", "output", "cpu"),
        ]
    ) as (y1, y2, y3, y4):
        with ir.For("i", 0, 4, nid="L1") as i:
            ir.MarkNid("S1")
            y1[i] = i + 1
            ir.MarkNid("S2")
            y2[i] = i + 2
            ir.MarkNid("S3")
            y3[i] = i + 3
            ir.MarkNid("S4")
            y4[i] = i + 4
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.swap(["S4", "S1"])
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_dependency():
    with ir.VarDef(
        [("y1", (4,), "int32", "inout", "cpu"), ("y2", (4,), "int32", "output", "cpu")]
    ) as (y1, y2):
        with ir.For("i", 0, 4, nid="L1") as i:
            ir.MarkNid("S1")
            y1[i] = i + 1
            ir.MarkNid("S2")
            y2[i] = y1[i] * 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.swap(["S2", "S1"])
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)
