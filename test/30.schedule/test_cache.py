import freetensor as ft
import pytest


def test_cache_read():
    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            y[i] = 0
            with ft.For("j", 0, 8, nid="L2") as j:
                ft.MarkNid("S0")
                y[i] = y[i] + x[i, j] * (x[i, j] + 1)
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.cache("S0", "x", "cpu")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = 0
            with ft.For("j", 0, 8) as j:
                with ft.VarDef("b", (1, 1), "int32", "cache", "cpu") as b:
                    b[0, 0] = x[i, j]
                    y[i] = y[i] + b[0, 0] * (b[0, 0] + 1)
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_cache_write():
    with ft.VarDef([
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            y[i] = 0
            with ft.For("j", 0, 8, nid="L2") as j:
                y[i] += x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.cache(s.find("L1").body, "y", "cpu")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("b", (1,), "int32", "cache", "cpu") as b:
                b[0] = 0
                with ft.For("j", 0, 8) as j:
                    b[0] += x[i, j] * 2
                y[i] = b[0]
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_reduction():
    with ft.VarDef([("x", (4, 8, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                with ft.For("k", 0, 8, nid="L3") as k:
                    y[i, j] = y[i, j] + x[i, j, k] * 2
    ast = ft.make_reduction(ft.pop_ast())
    print(ast)
    s = ft.Schedule(ast)
    s.cache("L2", "y", "cpu")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4, 8, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "inout", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("b", (1, 8), "int32", "cache", "cpu") as b:
                with ft.For("j1", 0, 8) as j:
                    b[0, j] = y[i, j]
                with ft.For("j", 0, 8) as j:
                    with ft.For("k", 0, 8) as k:
                        b[0, j] += x[i, j, k] * 2
                with ft.For("j1", 0, 8) as j:
                    y[i, j] = b[0, j]
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_cache_read_and_write():
    with ft.VarDef([
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "inout", "cpu"),
        ("z", (4, 8), "int32", "inout", "cpu"),
    ]) as (x, y, z):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                with ft.NamedScope("S0"):
                    z[i, j] = y[i, j] * 2
                    y[i, j] = x[i, j] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.cache("S0", "y", "cpu")
    ast = s.ast()
    print(ast)

    with ft.VarDef([
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "inout", "cpu"),
        ("z", (4, 8), "int32", "inout", "cpu"),
    ]) as (x, y, z):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 8) as j:
                with ft.VarDef("b", (1, 1), "int32", "cache", "cpu") as b:
                    b[0, 0] = y[i, j]
                    z[i, j] = b[0, 0] * 2
                    b[0, 0] = x[i, j] + 1
                    y[i, j] = b[0, 0]
    std = ft.pop_ast()

    assert std.match(ast)


def test_different_indices():
    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            ft.MarkNid("S0")
            y[i] = x[i] + x[i + 1]
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.cache("S0", "x", "cpu")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (5,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("b", (2,), "int32", "cache", "cpu") as b:
                with ft.For("j", i, i + 2) as j:
                    b[j + -1 * i] = x[j]
                y[i] = b[0] + b[1]
    std = ft.pop_ast()
    print(std)

    assert std.match(ast)


def test_no_var():
    with ft.VarDef([
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                ft.MarkNid("S0")
                y[i, j] = x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.cache("S0", "z", "cpu")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_no_stmt():
    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            y[i] = 0
            with ft.For("j", 0, 8, nid="L2") as j:
                y[i] = y[i] + x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.cache("S0", "x", "cpu")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_sharing_locals():
    with ft.VarDef([
        ("x", (4, 8), "int32", "input", "gpu/global"),
        ("y", (4, 8), "int32", "output", "gpu/global"),
    ]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2") as j:
                y[i, j] = x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.parallelize("L2", "threadIdx.x")
    ast = s.ast()
    with pytest.raises(ft.InvalidSchedule):
        s.cache("L2", "y", "gpu/local")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_local_var_as_index():
    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            y[i] = 0
            with ft.For("j", 0, 8, nid="L2") as j:
                y[i] = y[i] + x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.cache("L2", "x", "cpu")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=['prop_one_time_use'], verbose=1)

    with ft.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = 0
            with ft.VarDef("b", (1, 8), "int32", "cache", "cpu") as b:
                with ft.For("j1", 0, 8) as j:
                    b[0, j] = x[i, j]
                with ft.For("j2", 0, 8) as j:
                    y[i] = y[i] + b[0, j] * 2
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_cache_with_condition():
    with ft.VarDef([
        ("n", (), "int32", "input", "cpu"),
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (n, x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            y[i] = 0
            with ft.For("j", 0, 8, nid="L2") as j:
                with ft.If(n[()] > 0):
                    y[i] = y[i] + x[i, j] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.cache("L2", "x", "cpu")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("n", (), "int32", "input", "cpu"),
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (n, x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = 0
            with ft.If(n[()] > 0):
                with ft.VarDef("b", (1, 8), "int32", "cache", "cpu") as b:
                    with ft.For("j1", 0, 8) as j:
                        b[0, j] = x[i, j]
                    with ft.For("j2", 0, 8) as j:
                        y[i] = y[i] + b[0, j] * 2
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_cache_with_multiple_conditions():
    with ft.VarDef([
        ("n", (), "int32", "input", "cpu"),
        ("m", (), "int32", "input", "cpu"),
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (n, m, x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            y[i] = 0
            with ft.For("j", 0, 8, nid="L2") as j:
                with ft.If(n[()] > 0):
                    with ft.If(m[()] > 0):
                        y[i] = y[i] + x[i, j] * 2
                with ft.If(n[()] < 0):
                    y[i] = y[i] + x[i, j] * 3
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.cache("L2", "x", "cpu")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("n", (), "int32", "input", "cpu"),
        ("m", (), "int32", "input", "cpu"),
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (n, m, x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = 0
            with ft.VarDef("b", (1, 8), "int32", "cache", "cpu") as b:
                with ft.If(ft.l_or(ft.l_and(n[()] > 0, m[()] > 0), n[()] < 0)):
                    with ft.For("j1", 0, 8) as j:
                        b[0, j] = x[i, j]
                with ft.For("j2", 0, 8) as j:
                    with ft.If(n[()] > 0):
                        with ft.If(m[()] > 0):
                            y[i] = y[i] + b[0, j] * 2
                    with ft.If(n[()] < 0):
                        y[i] = y[i] + b[0, j] * 3
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_fill_is_necessary_when_possibly_not_written():
    with ft.VarDef([("x", (2, 4), "int32", "input", "cpu"),
                    ("y", (2, 4), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 2, nid="L1") as i:
            with ft.For("j", 0, 4, nid="L2") as j:
                with ft.If(i == 0):
                    y[0, j] = x[0, j]
                y[1, j] = x[1, j]
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.cache(s.find("L2").body, "y", "cpu")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (2, 4), "int32", "input", "cpu"),
                    ("y", (2, 4), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 2, nid="L1") as i:
            with ft.For("j", 0, 4, nid="L2") as j:
                with ft.VarDef("b", (2, 1), "int32", "cache", "cpu") as b:
                    with ft.For("k", 0, 2) as k:
                        b[k, 0] = y[k, j]  # This statement is necessary
                    with ft.If(i == 0):
                        b[0, 0] = x[0, j]
                    b[1, 0] = x[1, j]
                    with ft.For("k", 0, 2) as k:
                        y[k, j] = b[k, 0]
    std = ft.pop_ast()

    assert std.match(ast)
