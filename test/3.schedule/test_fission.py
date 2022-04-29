import freetensor as ft
import pytest


def test_fission_after():
    with ft.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                ft.MarkNid("S0")
                y[i, j] = i + j
                z[i, j] = i * j
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L2", ft.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 8) as j:
                y[i, j] = i + j
            with ft.For("j", 0, 8) as j:
                z[i, j] = i * j
    std = ft.pop_ast()

    assert std.match(ast)


def test_fission_before():
    with ft.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                y[i, j] = i + j
                ft.MarkNid("S0")
                z[i, j] = i * j
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L2", ft.FissionSide.Before, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 8) as j:
                y[i, j] = i + j
            with ft.For("j", 0, 8) as j:
                z[i, j] = i * j
    std = ft.pop_ast()

    assert std.match(ast)


def test_fission_after_empty():
    with ft.VarDef("z", (4, 8), "int32", "output", "cpu") as z:
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                ft.MarkNid("S0")
                z[i, j] = i * j
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L2", ft.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.simplify_pass(ast)
    print(ast)

    with ft.VarDef("z", (4, 8), "int32", "output", "cpu") as z:
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 8) as j:
                z[i, j] = i * j
    std = ft.pop_ast()

    assert std.match(ast)


def test_fission_before_empty():
    with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                ft.MarkNid("S0")
                y[i, j] = i + j
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L2", ft.FissionSide.Before, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.simplify_pass(ast)
    print(ast)

    with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 8) as j:
                y[i, j] = i + j
    std = ft.pop_ast()

    assert std.match(ast)


def test_stmt_in_if():
    with ft.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                with ft.If(i > 1):
                    ft.MarkNid("S0")
                    y[i, j] = i + j
                z[i, j] = i * j
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L2", ft.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ft.For("i", 0, 4) as i:
            with ft.If(i > 1):
                with ft.For("j", 0, 8) as j:
                    y[i, j] = i + j
            with ft.For("j", 0, 8) as j:
                z[i, j] = i * j
    std = ft.pop_ast()

    assert std.match(ast)


def test_buffer_hoist():
    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                with ft.VarDef("buf", (8,), "int32", "cache", "cpu") as b:
                    ft.MarkNid("S0")
                    b[j] = x0[i, j] + x1[i, j]
                    y[i, j] = b[j] * b[j]
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L2", ft.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("buf", (8,), "int32", "cache", "cpu") as b:
                with ft.For("j", 0, 8) as j:
                    b[j] = x0[i, j] + x1[i, j]
                with ft.For("j", 0, 8) as j:
                    y[i, j] = b[j] * b[j]
    std = ft.pop_ast()

    assert std.match(ast)


def test_buffer_no_hoist():
    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y, z):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                with ft.VarDef("buf", (4, 8), "int32", "cache", "cpu") as b:
                    b[i, j] = x0[i, j] + x1[i, j]
                    ft.MarkNid("S0")
                    y[i, j] = b[i, j] * b[i, j]
                    z[i, j] = x0[i, j] * 2
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L2", ft.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y, z):
        with ft.For("i", 0, 4) as i:
            # buf is not here
            with ft.For("j", 0, 8) as j:
                ft.Any()  # May be shrinked
            with ft.For("j", 0, 8) as j:
                z[i, j] = x0[i, j] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_correct_dependency_after():
    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                with ft.VarDef("buf", (1,), "int32", "cache", "cpu") as b:
                    ft.MarkNid("S0")
                    b[0] = x0[i, j] + x1[i, j]
                    y[i, j] = b[0] * b[0]
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L2", ft.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("buf", (8, 1), "int32", "cache", "cpu") as b:
                with ft.For("j", 0, 8) as j:
                    b[j, 0] = x0[i, j] + x1[i, j]
                with ft.For("j", 0, 8) as j:
                    y[i, j] = b[j, 0] * b[j, 0]
    std = ft.pop_ast()

    assert std.match(ast)


def test_correct_dependency_before():
    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                with ft.VarDef("buf", (1,), "int32", "cache", "cpu") as b:
                    b[0] = x0[i, j] + x1[i, j]
                    ft.MarkNid("S0")
                    y[i, j] = b[0] * b[0]
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L2", ft.FissionSide.Before, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("buf", (8, 1), "int32", "cache", "cpu") as b:
                with ft.For("j", 0, 8) as j:
                    b[j, 0] = x0[i, j] + x1[i, j]
                with ft.For("j", 0, 8) as j:
                    y[i, j] = b[j, 0] * b[j, 0]
    std = ft.pop_ast()

    assert std.match(ast)


def test_correct_dependency_loop_step():
    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 7, 2, nid="L2") as j:
                with ft.VarDef("buf", (1,), "int32", "cache", "cpu") as b:
                    ft.MarkNid("S0")
                    b[0] = x0[i, j] + x1[i, j]
                    y[i, j] = b[0] * b[0]
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L2", ft.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("buf", (3, 1), "int32", "cache", "cpu") as b:
                with ft.For("j", 0, 7, 2) as j:
                    b[j // 2, 0] = x0[i, j] + x1[i, j]
                with ft.For("j", 0, 7, 2) as j:
                    y[i, j] = b[j // 2, 0] * b[j // 2, 0]
    std = ft.use_builtin_div(ft.pop_ast())

    assert std.match(ast)


def test_correct_dependency_multi_loop_1():
    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                with ft.VarDef("buf", (1,), "int32", "cache", "cpu") as b:
                    ft.MarkNid("S0")
                    b[0] = x0[i, j] + x1[i, j]
                    y[i, j] = b[0] * b[0]
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L1", ft.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.VarDef("buf", (4, 8, 1), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 4) as i:
                with ft.For("j", 0, 8) as j:
                    b[i, j, 0] = x0[i, j] + x1[i, j]
            with ft.For("i", 0, 4) as i:
                with ft.For("j", 0, 8) as j:
                    y[i, j] = b[i, j, 0] * b[i, j, 0]
    std = ft.pop_ast()

    assert std.match(ast)


def test_correct_dependency_multi_loop_2():
    with ft.VarDef([("a", (4, 4), "float32", "input", "cpu"),
                    ("b", (4,), "float32", "input", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu"),
                    ("d_a", (4, 4), "float32", "inout", "cpu"),
                    ("d_b", (4,), "float32", "inout", "cpu")]) as (a, b, d_y,
                                                                   d_a, d_b):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 4) as j:
                with ft.VarDef("d_y_old", (), "float32", "cache",
                               "cpu") as d_y_old:
                    d_y_old[()] = d_y[i]
                    d_y[i] = 2 * d_y_old[()]
                    ft.MarkNid("S0")
                    d_a[i, j] += d_y_old[()] * b[j]
                    d_b[j] += d_y_old[()] * a[i, j]
            d_y[i] = 0
    ast = ft.make_reduction(ft.pop_ast())
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L1", ft.FissionSide.Before, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([("a", (4, 4), "float32", "input", "cpu"),
                    ("b", (4,), "float32", "input", "cpu"),
                    ("d_y", (4,), "float32", "inout", "cpu"),
                    ("d_a", (4, 4), "float32", "inout", "cpu"),
                    ("d_b", (4,), "float32", "inout", "cpu"),
                    ("d_y_old", (4, 4), "float32", "cache", "cpu")
                   ]) as (a, b, d_y, d_a, d_b, d_y_old):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 4) as j:
                d_y_old[i, j] = d_y[i]
                d_y[i] = 2 * d_y_old[i, j]
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 4) as j:
                d_a[i, j] += d_y_old[i, j] * b[j]
                d_b[j] += d_y_old[i, j] * a[i, j]
            d_y[i] = 0
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_correct_dependency_real_dep():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.VarDef("buf", (1,), "int32", "cache", "cpu") as b:
                ft.MarkNid("S0")
                b[0] = x[i] * 2
                with ft.For("j", 0, 8, nid="L2") as j:
                    y[i, j] = b[0] * b[0]
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L1", ft.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("buf", (4, 1), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 4) as i:
                b[i, 0] = x[i] * 2
            with ft.For("i", 0, 4) as i:
                with ft.For("j", 0, 8) as j:
                    y[i, j] = b[i, 0] * b[i, 0]
    std = ft.pop_ast()

    assert std.match(ast)


def test_correct_dependency_unable_resolve():
    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
        ("buf", (1,), "int32", "inout", "cpu"),
    ]) as (x0, x1, y, b):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                ft.MarkNid("S0")
                b[0] = x0[i, j] + x1[i, j]
                y[i, j] = b[0] * b[0]
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.fission("L2", ft.FissionSide.After, "S0")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_correct_dependency_no_need_to_modify_no_dep():
    with ft.VarDef([
        ("x0", (4, 4), "int32", "input", "cpu"),
        ("x1", (4, 4), "int32", "input", "cpu"),
        ("y", (4, 4, 4), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 4, nid="L2") as j:
                with ft.VarDef("buf", (4,), "int32", "cache", "cpu") as b:
                    with ft.For("k", 0, 4, nid="L3") as k:
                        ft.MarkNid("S0")
                        b[k] = x0[i, k]
                        y[i, j, k] = b[k] * x1[i, j]
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L2", ft.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("x0", (4, 4), "int32", "input", "cpu"),
        ("x1", (4, 4), "int32", "input", "cpu"),
        ("y", (4, 4, 4), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("buf", (4,), "int32", "cache", "cpu") as b:
                with ft.For("k", 0, 4) as k:
                    b[k] = x0[i, k]
                with ft.For("j", 0, 4) as j:
                    with ft.For("k", 0, 4) as k:
                        y[i, j, k] = b[k] * x1[i, j]
    std = ft.pop_ast()

    assert std.match(ast)


def test_correct_dependency_no_need_to_modify_broadcast():
    with ft.VarDef([
        ("x0", (4,), "int32", "input", "cpu"),
        ("x1", (4, 4), "int32", "input", "cpu"),
        ("y", (4, 4, 4), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 4, nid="L2") as j:
                with ft.VarDef("buf", (), "int32", "cache", "cpu") as b:
                    ft.MarkNid("S0")
                    b[()] = x0[i]
                    with ft.For("k", 0, 4, nid="L3") as k:
                        y[i, j, k] = b[()] * x1[i, j]
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L2", ft.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("x0", (4,), "int32", "input", "cpu"),
        ("x1", (4, 4), "int32", "input", "cpu"),
        ("y", (4, 4, 4), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("buf", (), "int32", "cache", "cpu") as b:
                b[()] = x0[i]
                with ft.For("j", 0, 4) as j:
                    with ft.For("k", 0, 4) as k:
                        y[i, j, k] = b[()] * x1[i, j]
    std = ft.pop_ast()

    assert std.match(ast)


def test_correct_dependency_overwritten_store():
    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                with ft.VarDef("buf", (1,), "int32", "cache", "cpu") as b:
                    b[0] = 1  # (1)
                    ft.MarkNid("S0")
                    with ft.If(j > 1):
                        b[0] += x0[i, j] + x1[i, j]  # (2)
                    y[i, j] = b[0] * b[0]  # (3)
    # Explanation: (3)->(1) is a real dependency, while (3)->(2) is not.
    # We cannot determine b is loop-invarient just becase b[0] = 1
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fission("L2", ft.FissionSide.After, "S0")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("x0", (4, 8), "int32", "input", "cpu"),
        ("x1", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x0, x1, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("buf", (8, 1), "int32", "cache", "cpu") as b:
                with ft.For("j", 0, 8) as j:
                    b[j, 0] = 1
                    with ft.If(j > 1):
                        b[j, 0] = x0[i, j] + x1[i, j] + 1
                with ft.For("j", 0, 8) as j:
                    y[i, j] = b[j, 0] * b[j, 0]
    std = ft.pop_ast()

    assert std.match(ast)
