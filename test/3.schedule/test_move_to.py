import ir
import pytest


def test_pure_swap_forward():
    with ir.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ir.For("i", 0, 4, nid="L1") as i:
            ir.MarkNid("S1")
            y1[i] = i + 1
            y2[i] = i + 2
            y3[i] = i + 3
            ir.MarkNid("S2")
            y4[i] = i + 4
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.move_to("S2", ir.MoveToSide.After, "S1")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ir.For("i", 0, 4) as i:
            y1[i] = i + 1
            y4[i] = i + 4
            y2[i] = i + 2
            y3[i] = i + 3
    std = ir.pop_ast()

    assert std.match(ast)


def test_pure_swap_backward():
    with ir.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ir.For("i", 0, 4, nid="L1") as i:
            ir.MarkNid("S1")
            y1[i] = i + 1
            y2[i] = i + 2
            y3[i] = i + 3
            ir.MarkNid("S2")
            y4[i] = i + 4
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.move_to("S1", ir.MoveToSide.Before, "S2")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ir.For("i", 0, 4) as i:
            y2[i] = i + 2
            y3[i] = i + 3
            y1[i] = i + 1
            y4[i] = i + 4
    std = ir.pop_ast()

    assert std.match(ast)


def test_swap_to_begin():
    with ir.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ir.For("i", 0, 4, nid="L1") as i:
            y1[i] = i + 1
            y2[i] = i + 2
            y3[i] = i + 3
            ir.MarkNid("S1")
            y4[i] = i + 4
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    dst = s.find(lambda x: x.nid() == "L1").node().body.stmts[0]
    s.move_to("S1", ir.MoveToSide.Before, dst)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ir.For("i", 0, 4) as i:
            y4[i] = i + 4
            y1[i] = i + 1
            y2[i] = i + 2
            y3[i] = i + 3
    std = ir.pop_ast()

    assert std.match(ast)


def test_swap_to_end():
    with ir.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ir.For("i", 0, 4, nid="L1") as i:
            ir.MarkNid("S1")
            y1[i] = i + 1
            y2[i] = i + 2
            y3[i] = i + 3
            y4[i] = i + 4
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    dst = s.find(lambda x: x.nid() == "L1").node().body.stmts[-1]
    s.move_to("S1", ir.MoveToSide.After, dst)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ir.For("i", 0, 4) as i:
            y2[i] = i + 2
            y3[i] = i + 3
            y4[i] = i + 4
            y1[i] = i + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_pure_fission_forward():
    with ir.VarDef([
        ("y1", (4, 4), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
    ]) as (y1, y2):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 4, nid="L2") as j:
                ir.MarkNid("S1")
                y1[i, j] = i * j + 1
                y2[i, j] = i * j + 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.move_to("S1", ir.MoveToSide.Before, "L2")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y1", (4, 4), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
    ]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                y1[i, j] = i * j + 1
            with ir.For("j", 0, 4) as j:
                y2[i, j] = i * j + 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_pure_fission_backward():
    with ir.VarDef([
        ("y1", (4, 4), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
    ]) as (y1, y2):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 4, nid="L2") as j:
                y1[i, j] = i * j + 1
                ir.MarkNid("S1")
                y2[i, j] = i * j + 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.move_to("S1", ir.MoveToSide.After, "L2")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y1", (4, 4), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
    ]) as (y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                y1[i, j] = i * j + 1
            with ir.For("j", 0, 4) as j:
                y2[i, j] = i * j + 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_swap_and_fission_forward():
    with ir.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
        ("y3", (4, 4), "int32", "output", "cpu"),
    ]) as (y1, y2, y3):
        with ir.For("i", 0, 4, nid="L1") as i:
            ir.MarkNid("S1")
            y1[i] = i
            with ir.For("j", 0, 4, nid="L2") as j:
                ir.MarkNid("S2")
                y2[i, j] = i * j + 1
                ir.MarkNid("S3")
                y3[i, j] = i * j + 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.move_to("S3", ir.MoveToSide.Before, "S1")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
        ("y3", (4, 4), "int32", "output", "cpu"),
    ]) as (y1, y2, y3):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                y3[i, j] = i * j + 2
            y1[i] = i
            with ir.For("j", 0, 4) as j:
                y2[i, j] = i * j + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_swap_and_fission_backward():
    with ir.VarDef([
        ("y1", (4, 4), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 4, nid="L2") as j:
                ir.MarkNid("S2")
                y1[i, j] = i * j + 1
                ir.MarkNid("S3")
                y2[i, j] = i * j + 2
            ir.MarkNid("S1")
            y3[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.move_to("S2", ir.MoveToSide.After, "S1")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y1", (4, 4), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                y2[i, j] = i * j + 2
            y3[i] = i
            with ir.For("j", 0, 4) as j:
                y1[i, j] = i * j + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_crossing_var_def():
    with ir.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ir.For("i", 0, 4, nid="L1") as i:
            y1[i] = i
            with ir.For("j", 0, 4, nid="L2") as j:
                with ir.VarDef("t", (), "int32", "cache", "cpu") as t:
                    ir.MarkNid("S1")
                    t[()] = x[i] * x[j]
                    y2[i, j] = t[()]
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    dst = s.find(lambda x: x.nid() == "L1").node().body.stmts[0]
    s.move_to("S1", ir.MoveToSide.Before, dst)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("t", (4,), "int32", "cache", "cpu") as t:
                with ir.For("j", 0, 4) as j:
                    t[j] = x[i] * x[j]
                y1[i] = i
                with ir.For("j", 0, 4) as j:
                    y2[i, j] = t[j]
    std = ir.pop_ast()

    assert std.match(ast)
