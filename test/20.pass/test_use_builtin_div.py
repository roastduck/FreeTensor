import freetensor as ft
import pytest


def test_ge0_floor_div_ge0():
    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        with ft.Assert(a[()] >= 0):
            with ft.Assert(b[()] >= 0):
                c[()] = a[()] // b[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        with ft.Assert(a[()] >= 0):
            with ft.Assert(b[()] >= 0):
                c[()] = ft.round_towards_0_div(a[()], b[()])
    std = ft.pop_ast()

    assert std.match(ast)


def test_unknown_floor_div_unknown():
    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        c[()] = a[()] // b[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        c[()] = a[()] // b[()]
    std = ft.pop_ast()

    assert std.match(ast)


def test_ge0_ceil_div_ge0():
    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        with ft.Assert(a[()] >= 0):
            with ft.Assert(b[()] >= 0):
                c[()] = ft.ceildiv(a[()], b[()])
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        with ft.Assert(a[()] >= 0):
            with ft.Assert(b[()] >= 0):
                c[()] = ft.round_towards_0_div(a[()] + (b[()] - 1), b[()])
    std = ft.pop_ast()

    assert std.match(ast)


def test_unknown_ceil_div_unknown():
    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        c[()] = ft.ceildiv(a[()], b[()])
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        c[()] = ft.ceildiv(a[()], b[()])
    std = ft.pop_ast()

    assert std.match(ast)


def test_ge0_mod_ge0():
    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        with ft.Assert(a[()] >= 0):
            with ft.Assert(b[()] >= 0):
                c[()] = a[()] % b[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        with ft.Assert(a[()] >= 0):
            with ft.Assert(b[()] >= 0):
                c[()] = ft.remainder(a[()], b[()])
    std = ft.pop_ast()

    assert std.match(ast)


def test_mod_mod_ge0():
    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "input", "cpu"),
        ("d", (), "int32", "output", "cpu"),
    ]) as (a, b, c, d):
        with ft.Assert(a[()] >= 0):
            with ft.Assert(b[()] >= 0):
                with ft.Assert(c[()] >= 0):
                    d[()] = (a[()] % b[()]) % c[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "input", "cpu"),
        ("d", (), "int32", "output", "cpu"),
    ]) as (a, b, c, d):
        with ft.Assert(a[()] >= 0):
            with ft.Assert(b[()] >= 0):
                with ft.Assert(c[()] >= 0):
                    d[()] = ft.remainder(ft.remainder(a[()], b[()]), c[()])
    std = ft.pop_ast()

    assert std.match(ast)


def test_unknown_mod_unknown():
    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        c[()] = a[()] % b[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        c[()] = a[()] % b[()]
    std = ft.pop_ast()

    assert std.match(ast)
