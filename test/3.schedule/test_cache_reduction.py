import ir
import pytest


def test_reduce_sum():
    with ir.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "inout", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                y[i, j] = y[i, j] + x[i, j] * 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.cache_reduction("L2", "y", "cpu")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "inout", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("b", (1, 8), "int32", "cache", "cpu") as b:
                with ir.For("j", 0, 8) as j:
                    b[0, j] = 2 * x[i, j]  # After remove_writes pass
                with ir.For("j1", 0, 8) as j:
                    y[i, j] += b[0, j]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_reduce_sum_loop():
    with ir.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "inout", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                y[i] = y[i] + x[i, j] * 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.cache_reduction("L2", "y", "cpu")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "inout", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("b", (1,), "int32", "cache", "cpu") as b:
                b[0] = 0
                with ir.For("j2", 0, 8) as j:
                    b[0] = b[0] + x[i, j] * 2
                y[i] = y[i] + b[0]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_reduce_min_loop():
    with ir.VarDef([
        ("x", (4, 8), "float32", "input", "cpu"),
        ("y", (4,), "float32", "inout", "cpu"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                y[i] = ir.min(y[i], x[i, j] * 2)
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.cache_reduction("L2", "y", "cpu")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x", (4, 8), "float32", "input", "cpu"),
        ("y", (4,), "float32", "inout", "cpu"),
    ]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.VarDef("b", (1,), "float32", "cache", "cpu") as b:
                b[0] = float("inf")
                with ir.For("j2", 0, 8) as j:
                    b[0] = ir.min(b[0], x[i, j] * 2)
                y[i] = ir.min(y[i], b[0])
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_no_var():
    with ir.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "inout", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                ir.MarkNid("S0")
                y[i, j] = y[i, j] + x[i, j] * 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.cache_reduction("S0", "z", "cpu")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_no_stmt():
    with ir.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "inout", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                y[i, j] = y[i, j] + x[i, j] * 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.cache_reduction("S0", "x", "cpu")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_read_not_allowed():
    with ir.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            y[i] = 0
            with ir.For("j", 0, 8, nid="L2") as j:
                ir.MarkNid("S0")
                y[i] = y[i] + x[i, j] * 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.cache_reduction("S0", "x", "cpu")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_write_not_allowed():
    with ir.VarDef([
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                ir.MarkNid("S0")
                y[i, j] = x[i, j] * 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.cache_reduction("S0", "y", "cpu")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_mix_op_not_allowed():
    with ir.VarDef([("x", (4, 8), "int32", "input", "cpu"),
                    ("y", (4, 8), "int32", "inout", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2") as j:
                with ir.NamedScope("S0"):
                    y[i, j] = y[i, j] + x[i, j] * 2
                    y[i, j] = ir.min(y[i, j], x[i, j] * 2)
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.cache_reduction("S0", "y", "cpu")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)
