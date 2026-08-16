import freetensor as ft
import pytest


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_basic(as_subprocess):
    with ft.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ft.For("i", 0, 4, label="L1") as i:
            ft.MarkLabel("S1")
            y1[i] = i + 1
            ft.MarkLabel("S2")
            y2[i] = i + 2
            ft.MarkLabel("S3")
            y3[i] = i + 3
            ft.MarkLabel("S4")
            y4[i] = i + 4
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.swap(["S2", "S3", "S1"], as_subprocess=as_subprocess)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

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


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_not_consecutive(as_subprocess):
    with ft.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ft.For("i", 0, 4, label="L1") as i:
            ft.MarkLabel("S1")
            y1[i] = i + 1
            ft.MarkLabel("S2")
            y2[i] = i + 2
            ft.MarkLabel("S3")
            y3[i] = i + 3
            ft.MarkLabel("S4")
            y4[i] = i + 4
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.swap(["S4", "S1"], as_subprocess=as_subprocess)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_dependence(as_subprocess):
    with ft.VarDef([("y1", (4,), "int32", "inout", "cpu"),
                    ("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
        with ft.For("i", 0, 4, label="L1") as i:
            ft.MarkLabel("S1")
            y1[i] = i + 1
            ft.MarkLabel("S2")
            y2[i] = y1[i] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.swap(["S2", "S1"], as_subprocess=as_subprocess)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_legal_dependence(as_subprocess):
    with ft.VarDef([("x1", (1000,), "int32", "input", "cpu"),
                    ("x2", (1000,), "int32", "input", "cpu"),
                    ("y", (1001,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ft.For("i", 0, 1001) as i:
            y[i] = 0
        with ft.For("i", 0, 1000) as i:
            ft.MarkLabel("S1")
            y[i] += x1[i]  # Always happens afterhand swapped or not
            ft.MarkLabel("S2")
            y[i + 1] = x2[i]  # Always happens beforehand swapped or not
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.swap(["S2", "S1"], as_subprocess=as_subprocess)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (1000,), "int32", "input", "cpu"),
                    ("x2", (1000,), "int32", "input", "cpu"),
                    ("y", (1001,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ft.For("i", 0, 1001) as i:
            y[i] = 0
        with ft.For("i", 0, 1000) as i:
            y[i + 1] = x2[i]
            y[i] += x1[i]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_crossing_var_def(as_subprocess):
    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4, 4), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.For("i", 0, 4, label="L1") as i:
            ft.MarkLabel("S1")
            y1[i] = i
            with ft.VarDef("t", (4,), "int32", "cache", "cpu") as t:
                with ft.For("j", 0, 4, label="L2") as j:
                    t[j] = x[i] * x[j]
                with ft.For("j", 0, 4, label="L3") as j:
                    y2[i, j] = t[j]
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.swap(["L2", "S1"], as_subprocess=as_subprocess)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, skip_passes=["prop_one_time_use"], verbose=1)

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


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_select_multiple(as_subprocess):
    with ft.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ft.For("i", 0, 4, label="L1") as i:
            ft.MarkLabel("S1")
            y1[i] = i + 1
            ft.MarkLabel("S2")
            y2[i] = i + 2
            ft.MarkLabel("S3")
            y3[i] = i + 3
            ft.MarkLabel("S4")
            y4[i] = i + 4
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.swap(["S3|S4", "S1|S2"], as_subprocess=as_subprocess)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
    ]) as (y1, y2, y3, y4):
        with ft.For("i", 0, 4) as i:
            y3[i] = i + 3
            y4[i] = i + 4
            y1[i] = i + 1
            y2[i] = i + 2
    std = ft.pop_ast()

    assert std.match(ast)
