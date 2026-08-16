import freetensor as ft
import pytest


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_reduce_add(as_subprocess):
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] = y[...] + x[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] += x[i]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_reduce_prod(as_subprocess):
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[...] = 1
        with ft.For("i", 0, 4) as i:
            y[...] = y[...] * x[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[...] = 1
        with ft.For("i", 0, 4) as i:
            y[...] *= x[i]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_reduce_sub(as_subprocess):
    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] = y[...] - x[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("x", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] += -1 * x[i]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_reduce_add_sub_1(as_subprocess):
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] = (y[...] - x1[i]) - x2[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] += -1 * (x1[i] + x2[i])
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_reduce_add_sub_2(as_subprocess):
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] = (y[...] - x1[i]) + x2[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] += x2[i] - x1[i]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_reduce_add_sub_3(as_subprocess):
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] = y[...] - (x1[i] - x2[i])
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] += x2[i] - x1[i]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_reduce_add_sub_4(as_subprocess):
    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] = y[...] - (x1[i] + x2[i])
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("x1", (4,), "float32", "input", "cpu"),
                    ("x2", (4,), "float32", "input", "cpu"),
                    ("y", (), "float32", "output", "cpu")]) as (x1, x2, y):
        y[...] = 0
        with ft.For("i", 0, 4) as i:
            y[...] += -1 * (x1[i] + x2[i])
    std = ft.pop_ast()

    assert std.match(ast)
