import freetensor as ft
import pytest


def test_pure_swap_forward():
    with ft.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ft.For("i", 0, 4, nid="L1") as i:
            ft.MarkNid("S1")
            y1[i] = i + 1
            y2[i] = i + 2
            y3[i] = i + 3
            ft.MarkNid("S2")
            y4[i] = i + 4
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.move_to("S2", ft.MoveToSide.After, "S1")
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
            y1[i] = i + 1
            y4[i] = i + 4
            y2[i] = i + 2
            y3[i] = i + 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_pure_swap_backward():
    with ft.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ft.For("i", 0, 4, nid="L1") as i:
            ft.MarkNid("S1")
            y1[i] = i + 1
            y2[i] = i + 2
            y3[i] = i + 3
            ft.MarkNid("S2")
            y4[i] = i + 4
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.move_to("S1", ft.MoveToSide.Before, "S2")
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


def test_swap_to_begin():
    with ft.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ft.For("i", 0, 4, nid="L1") as i:
            y1[i] = i + 1
            y2[i] = i + 2
            y3[i] = i + 3
            ft.MarkNid("S1")
            y4[i] = i + 4
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    dst = s.find("L1").body.stmts[0]
    s.move_to("S1", ft.MoveToSide.Before, dst)
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
            y4[i] = i + 4
            y1[i] = i + 1
            y2[i] = i + 2
            y3[i] = i + 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_swap_to_end():
    with ft.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ft.For("i", 0, 4, nid="L1") as i:
            ft.MarkNid("S1")
            y1[i] = i + 1
            y2[i] = i + 2
            y3[i] = i + 3
            y4[i] = i + 4
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    dst = s.find("L1").body.stmts[-1]
    s.move_to("S1", ft.MoveToSide.After, dst)
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
            y4[i] = i + 4
            y1[i] = i + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_pure_fission_forward():
    with ft.VarDef([
        ("y1", (4, 4), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
    ]) as (y1, y2):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 4, nid="L2") as j:
                ft.MarkNid("S1")
                y1[i, j] = i * j + 1
                y2[i, j] = i * j + 2
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.move_to("S1", ft.MoveToSide.Before, "L2")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("y1", (4, 4), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
    ]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 4) as j:
                y1[i, j] = i * j + 1
            with ft.For("j", 0, 4) as j:
                y2[i, j] = i * j + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_pure_fission_backward():
    with ft.VarDef([
        ("y1", (4, 4), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
    ]) as (y1, y2):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 4, nid="L2") as j:
                y1[i, j] = i * j + 1
                ft.MarkNid("S1")
                y2[i, j] = i * j + 2
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.move_to("S1", ft.MoveToSide.After, "L2")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("y1", (4, 4), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
    ]) as (y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 4) as j:
                y1[i, j] = i * j + 1
            with ft.For("j", 0, 4) as j:
                y2[i, j] = i * j + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_swap_and_fission_forward():
    with ft.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
        ("y3", (4, 4), "int32", "output", "cpu"),
    ]) as (y1, y2, y3):
        with ft.For("i", 0, 4, nid="L1") as i:
            ft.MarkNid("S1")
            y1[i] = i
            with ft.For("j", 0, 4, nid="L2") as j:
                ft.MarkNid("S2")
                y2[i, j] = i * j + 1
                ft.MarkNid("S3")
                y3[i, j] = i * j + 2
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.move_to("S3", ft.MoveToSide.Before, "S1")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
        ("y3", (4, 4), "int32", "output", "cpu"),
    ]) as (y1, y2, y3):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 4) as j:
                y3[i, j] = i * j + 2
            y1[i] = i
            with ft.For("j", 0, 4) as j:
                y2[i, j] = i * j + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_swap_and_fission_backward():
    with ft.VarDef([
        ("y1", (4, 4), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 4, nid="L2") as j:
                ft.MarkNid("S2")
                y1[i, j] = i * j + 1
                ft.MarkNid("S3")
                y2[i, j] = i * j + 2
            ft.MarkNid("S1")
            y3[i] = i
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.move_to("S2", ft.MoveToSide.After, "S1")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("y1", (4, 4), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 4) as j:
                y2[i, j] = i * j + 2
            y3[i] = i
            with ft.For("j", 0, 4) as j:
                y1[i, j] = i * j + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_crossing_var_def():
    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 4, nid="L1") as i:
            y1[i] = i
            with ft.For("j", 0, 4, nid="L2") as j:
                with ft.VarDef("t", (), "int32", "cache", "cpu") as t:
                    ft.MarkNid("S1")
                    t[()] = x[i] * x[j]
                    y2[i, j] = t[()]
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    dst = s.find("L1").body.stmts[0]
    s.move_to("S1", ft.MoveToSide.Before, dst)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
                with ft.For("j", 0, 4) as j:
                    t[j] = x[i] * x[j]
                y1[i] = i
                with ft.For("j", 0, 4) as j:
                    y2[i, j] = t[j]
    std = ft.pop_ast()

    assert std.match(ast)
