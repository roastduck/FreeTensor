import ir
import pytest


def test_basic():
    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j1", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
            with ir.For("j2", 0, 8, nid="L2b") as j:
                z[i, j] = i * j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fuse("L2a", "L2b")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 8) as j:
                y[i, j] = i + j
                z[i, j] = i * j
    std = ir.pop_ast()

    assert std.match(ast)


def test_not_aligned_1():
    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j1", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
            with ir.For("j2", 2, 10, nid="L2b") as j:
                z[i, j - 2] = i * (j - 2)
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fuse("L2a", "L2b")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 8) as j:
                y[i, j] = i + j
                z[i, j] = i * j
    std = ir.pop_ast()

    assert std.match(ast)


def test_not_aligned_2():
    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
            with ir.For("j", 2, 10, nid="L2b") as j:
                z[i, j - 2] = i * (j - 2)
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fuse("L2a", "L2b")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 8) as j:
                y[i, j] = i + j
                z[i, j] = i * j
    std = ir.pop_ast()

    assert std.match(ast)


def test_step():
    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j1", 0, 8, 2, nid="L2a") as j:
                y[i, j] = i + j
            with ir.For("j2", 1, 9, 2, nid="L2b") as j:
                z[i, j] = i * j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fuse("L2a", "L2b")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                y[i, j * 2] = i + j * 2
                z[i, j * 2 + 1] = i * (j * 2 + 1)
    std = ir.pop_ast()

    assert std.match(ast)


def test_not_following_1():
    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
        ("w", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z, w):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
            with ir.For("j", 0, 8, nid="L2b") as j:
                z[i, j] = i * j
            with ir.For("j", 0, 8, nid="L2c") as j:
                w[i, j] = i - j
    ast = ir.simplify_pass(ir.pop_ast())
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.fuse("L2a", "L2c")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_not_following_2():
    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4, nid="L1a") as i:
            with ir.For("j", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
        with ir.For("i", 0, 4, nid="L1b") as i:
            with ir.For("j", 0, 8, nid="L2b") as j:
                z[i, j] = i * j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.fuse("L2a", "L2b")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_different_length():
    with ir.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
            with ir.For("j", 0, 10, nid="L2b") as j:
                z[i, j] = i * j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.fuse("L2a", "L2b")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_dependency_unable_resolve():
    with ir.VarDef([
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.VarDef("b", (4, 8), "int32", "cache", "cpu") as b:
                with ir.For("j", 0, 8, nid="L2a") as j:
                    b[i, j] = x[i, j] * 2
                with ir.For("j", 0, 8, nid="L2b") as j:
                    y[i, j] = b[i, 8 - j]
    ast = ir.simplify_pass(ir.pop_ast())
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.fuse("L2a", "L2b")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_buffer_fuse():
    with ir.VarDef([
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y1", (4, 8), "int32", "output", "cpu"),
        ("y2", (4, 8), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.VarDef("b", (4, 8), "int32", "cache", "cpu") as b:
                with ir.For("j", 0, 8, nid="L2a") as j:
                    b[i, j] = x[i, j] * 2
                with ir.For("j", 0, 8, nid="L2b") as j:
                    y1[i, j] = b[i, j] + 1
                    y2[i, j] = b[i, j] + 2
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fuse("L2a", "L2b")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y1", (4, 8), "int32", "output", "cpu"),
        ("y2", (4, 8), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 8) as j:
                with ir.VarDef("b", (1, 1), "int32", "cache", "cpu") as b:
                    b[0, 0] = x[i, j] * 2
                    y1[i, j] = b[0, 0] + 1
                    y2[i, j] = b[0, 0] + 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_hoist_var():
    with ir.For("i", 0, 4, nid="L1") as i:
        with ir.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
            with ir.For("j1", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
        with ir.VarDef("z", (4, 8), "int32", "output", "cpu") as z:
            with ir.For("j2", 0, 8, nid="L2b") as j:
                z[i, j] = i * j
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fuse("L2a", "L2b")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.For("i", 0, 4) as i:
        with ir.VarDef([
            ("y", (4, 8), "int32", "output", "cpu"),
            ("z", (4, 8), "int32", "output", "cpu"),
        ]) as (y, z):
            with ir.For("j", 0, 8) as j:
                y[i, j] = i + j
                z[i, j] = i * j
    std = ir.pop_ast()

    assert std.match(ast)


def test_hoist_var_in_stmt_seq():
    with ir.For("i", 0, 4, nid="L1") as i:
        with ir.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
            with ir.For("j1", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
        with ir.VarDef("z", (4, 8), "int32", "output", "cpu") as z:
            with ir.For("j2", 0, 8, nid="L2b") as j:
                z[i, j] = i * j
            z[0, 0] = -1
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    s.fuse("L2a", "L2b")
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.For("i", 0, 4) as i:
        with ir.VarDef([
            ("y", (4, 8), "int32", "output", "cpu"),
            ("z", (4, 8), "int32", "output", "cpu"),
        ]) as (y, z):
            with ir.For("j", 0, 8) as j:
                y[i, j] = i + j
                z[i, j] = i * j
            z[0, 0] = -1
    std = ir.pop_ast()

    assert std.match(ast)


def test_hoist_var_with_modified_shape():
    with ir.VarDef("n", (), "int32", "output", "cpu") as n:
        with ir.For("i", 0, 4, nid="L1") as i:
            n[()] = i
        with ir.VarDef("x", (n,), "int32", "output", "cpu") as x:
            with ir.For("i", 0, 4, nid="L2") as i:
                x[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.fuse("L1", "L2")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)
