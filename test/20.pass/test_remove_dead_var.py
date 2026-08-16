import freetensor as ft
import pytest


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_basic(as_subprocess):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
        y[()] = x[()] + 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] + 1
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_chained(as_subprocess):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
            with ft.VarDef("b", (), "int32", "cache", "cpu") as b:
                b[()] = a[()] + 1
                with ft.VarDef("c", (), "int32", "cache", "cpu") as c:
                    c[()] = a[()] + b[()]
        y[()] = x[()] + 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] + 1
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_self_assign(as_subprocess):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = 0
            with ft.For("i", 0, 10) as i:
                a[()] = a[()] * x[()] + 1
        y[()] = x[()] + 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] + 1
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_remove_a_write_if_no_reads_after_it(as_subprocess):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
            y[()] = a[()] * a[()]
            a[()] *= 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
            y[()] = a[()] * a[()]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_remove_a_write_in_a_loop_if_no_reads_after_it(as_subprocess):
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (4,), "int32", "cache", "cpu") as a:
            with ft.For("i", 0, 4) as i:
                a[i] = x[i] + 1
            with ft.For("i", 0, 4) as i:
                y[i] = a[i] * a[i]
            with ft.For("i", 0, 4) as i:
                a[i] *= 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (4,), "int32", "cache", "cpu") as a:
            with ft.For("i", 0, 4) as i:
                a[i] = x[i] + 1
            with ft.For("i", 0, 4) as i:
                y[i] = a[i] * a[i]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_no_remove_writes_if_maybe_looped_around(as_subprocess):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (2,), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
            with ft.For("i", 0, 2) as i:
                y[i] = a[()] * a[()]
                a[()] *= 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (2,), "int32", "output", "cpu")]) as (x, y):
        with ft.VarDef("a", (), "int32", "cache", "cpu") as a:
            a[()] = x[()] + 1
            with ft.For("i", 0, 2) as i:
                y[i] = a[()] * a[()]
                a[()] *= 2
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_input_mutable_can_be_optimized_out(as_subprocess):
    with ft.VarDef([("x", (), "int32", "input-mutable", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] * 2
        x[()] += 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] * 2
    std = ft.pop_ast()

    assert std.match(ast)
