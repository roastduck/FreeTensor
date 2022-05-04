import freetensor as ft
import pytest


def test_basic():
    with ft.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j1", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
            with ft.For("j2", 0, 8, nid="L2b") as j:
                z[i, j] = i * j
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fuse("L2a", "L2b")
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
                z[i, j] = i * j
    std = ft.pop_ast()

    assert std.match(ast)


def test_not_aligned_1():
    with ft.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j1", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
            with ft.For("j2", 2, 10, nid="L2b") as j:
                z[i, j - 2] = i * (j - 2)
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fuse("L2a", "L2b")
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
                z[i, j] = i * j
    std = ft.pop_ast()

    assert std.match(ast)


def test_not_aligned_2():
    with ft.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
            with ft.For("j", 2, 10, nid="L2b") as j:
                z[i, j - 2] = i * (j - 2)
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fuse("L2a", "L2b")
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
                z[i, j] = i * j
    std = ft.pop_ast()

    assert std.match(ast)


def test_step():
    with ft.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j1", 0, 8, 2, nid="L2a") as j:
                y[i, j] = i + j
            with ft.For("j2", 1, 9, 2, nid="L2b") as j:
                z[i, j] = i * j
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fuse("L2a", "L2b")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 4) as j:
                y[i, j * 2] = i + j * 2
                z[i, j * 2 + 1] = i * (j * 2 + 1)
    std = ft.pop_ast()

    assert std.match(ast)


def test_not_following_1():
    with ft.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
        ("w", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z, w):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
            with ft.For("j", 0, 8, nid="L2b") as j:
                z[i, j] = i * j
            with ft.For("j", 0, 8, nid="L2c") as j:
                w[i, j] = i - j
    ast = ft.simplify_pass(ft.pop_ast())
    print(ast)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.fuse("L2a", "L2c")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_not_following_2():
    with ft.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ft.For("i", 0, 4, nid="L1a") as i:
            with ft.For("j", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
        with ft.For("i", 0, 4, nid="L1b") as i:
            with ft.For("j", 0, 8, nid="L2b") as j:
                z[i, j] = i * j
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.fuse("L2a", "L2b")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_different_length():
    with ft.VarDef([
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (y, z):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
            with ft.For("j", 0, 10, nid="L2b") as j:
                z[i, j] = i * j
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.fuse("L2a", "L2b")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_dependency_unable_resolve():
    with ft.VarDef([
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.VarDef("b", (4, 8), "int32", "cache", "cpu") as b:
                with ft.For("j", 0, 8, nid="L2a") as j:
                    b[i, j] = x[i, j] * 2
                with ft.For("j", 0, 8, nid="L2b") as j:
                    y[i, j] = b[i, 8 - j]
    ast = ft.simplify_pass(ft.pop_ast())
    print(ast)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.fuse("L2a", "L2b")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_buffer_fuse():
    with ft.VarDef([
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y1", (4, 8), "int32", "output", "cpu"),
        ("y2", (4, 8), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.VarDef("b", (4, 8), "int32", "cache", "cpu") as b:
                with ft.For("j", 0, 8, nid="L2a") as j:
                    b[i, j] = x[i, j] * 2
                with ft.For("j", 0, 8, nid="L2b") as j:
                    y1[i, j] = b[i, j] + 1
                    y2[i, j] = b[i, j] + 2
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fuse("L2a", "L2b")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("x", (4, 8), "int32", "input", "cpu"),
        ("y1", (4, 8), "int32", "output", "cpu"),
        ("y2", (4, 8), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 8) as j:
                with ft.VarDef("b", (1, 1), "int32", "cache", "cpu") as b:
                    b[0, 0] = x[i, j] * 2
                    y1[i, j] = b[0, 0] + 1
                    y2[i, j] = b[0, 0] + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_hoist_var():
    with ft.For("i", 0, 4, nid="L1") as i:
        with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
            with ft.For("j1", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
        with ft.VarDef("z", (4, 8), "int32", "output", "cpu") as z:
            with ft.For("j2", 0, 8, nid="L2b") as j:
                z[i, j] = i * j
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fuse("L2a", "L2b")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.For("i", 0, 4) as i:
        with ft.VarDef([
            ("y", (4, 8), "int32", "output", "cpu"),
            ("z", (4, 8), "int32", "output", "cpu"),
        ]) as (y, z):
            with ft.For("j", 0, 8) as j:
                y[i, j] = i + j
                z[i, j] = i * j
    std = ft.pop_ast()

    assert std.match(ast)


def test_hoist_var_in_stmt_seq():
    with ft.For("i", 0, 4, nid="L1") as i:
        with ft.VarDef("y", (4, 8), "int32", "output", "cpu") as y:
            with ft.For("j1", 0, 8, nid="L2a") as j:
                y[i, j] = i + j
        with ft.VarDef("z", (4, 8), "int32", "output", "cpu") as z:
            with ft.For("j2", 0, 8, nid="L2b") as j:
                z[i, j] = i * j
            z[0, 0] = -1
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fuse("L2a", "L2b")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.For("i", 0, 4) as i:
        with ft.VarDef([
            ("y", (4, 8), "int32", "output", "cpu"),
            ("z", (4, 8), "int32", "output", "cpu"),
        ]) as (y, z):
            with ft.For("j", 0, 8) as j:
                y[i, j] = i + j
                z[i, j] = i * j
            z[0, 0] = -1
    std = ft.pop_ast()

    assert std.match(ast)


def test_hoist_var_with_modified_shape():
    with ft.VarDef("n", (), "int32", "output", "cpu") as n:
        with ft.For("i", 0, 4, nid="L1") as i:
            n[()] = i
        with ft.VarDef("x", (n,), "int32", "output", "cpu") as x:
            with ft.For("i", 0, 4, nid="L2") as i:
                x[i] = i
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.fuse("L1", "L2")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_fuse_no_deps_1():

    @ft.transform
    def test(ptr, edge1, edge2):
        ptr: ft.Var[(11,), "int32", "input", "cpu"]
        edge1: ft.Var[(50,), "int32", "input", "cpu"]
        edge2: ft.Var[(50,), "int32", "output", "cpu"]
        'nid: Li1'
        'no_deps: edge2'
        for i in range(10):
            for j in range(ptr[i], ptr[i + 1]):
                edge2[j] = edge1[j] + i
        'nid: Li2'
        'no_deps: edge2'
        for i in range(10):
            for j in range(ptr[i], ptr[i + 1]):
                edge2[j] += j

    print(test)
    s = ft.Schedule(test)
    fused = s.fuse("Li1", "Li2")
    print(s.ast())
    assert s.find(fused).property.no_deps == ["edge2"]


def test_fuse_no_deps_2():

    @ft.transform
    def test(ptr, edge1, edge2):
        ptr: ft.Var[(11,), "int32", "input", "cpu"]
        edge1: ft.Var[(50,), "int32", "input", "cpu"]
        edge2: ft.Var[(50,), "int32", "output", "cpu"]
        foobar: ft.Var[(10,), "int32", "output", "cpu"]
        'nid: Li1'
        'no_deps: edge2'
        for i in range(10):
            for j in range(ptr[i], ptr[i + 1]):
                edge2[j] = edge1[j] + i
        'nid: Li2'
        for i in range(10):
            # Nothing to do with edge2 here
            foobar[i] = i

    print(test)
    s = ft.Schedule(test)
    fused = s.fuse("Li1", "Li2")
    print(s.ast())
    assert s.find(fused).property.no_deps == ["edge2"]


def test_fuse_no_deps_3():

    @ft.transform
    def test(ptr, edge1, edge2):
        ptr: ft.Var[(11,), "int32", "input", "cpu"]
        edge1: ft.Var[(50,), "int32", "input", "cpu"]
        edge2: ft.Var[(50,), "int32", "output", "cpu"]
        'nid: Li1'
        'no_deps: edge2'
        for i in range(10):
            for j in range(ptr[i], ptr[i + 1]):
                edge2[j] = edge1[j] + i
        'nid: Li2'  # If we don't mark edge2 here
        for i in range(10):
            for j in range(ptr[i], ptr[i + 1] + 1):
                edge2[j] += j

    print(test)
    s = ft.Schedule(test)
    with pytest.raises(ft.InvalidSchedule):
        s.fuse("Li1", "Li2")
    ast = s.ast()  # Should not changed
    assert ast.match(test.body)


def test_fuse_with_if():
    with ft.VarDef([
        ("c", (), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (c, y, z):
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.If(c[()] > 0):
                with ft.For("j1", 0, 8, nid="L2a") as j:
                    y[i, j] = i + j
            with ft.For("j2", 0, 8, nid="L2b") as j:
                z[i, j] = i * j
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.fuse("L2a", "L2b")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast)
    print(ast)

    with ft.VarDef([
        ("c", (), "int32", "input", "cpu"),
        ("y", (4, 8), "int32", "output", "cpu"),
        ("z", (4, 8), "int32", "output", "cpu"),
    ]) as (c, y, z):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 8) as j:
                with ft.If(c[()] > 0):
                    y[i, j] = i + j
                z[i, j] = i * j
    std = ft.pop_ast()

    assert std.match(ast)
