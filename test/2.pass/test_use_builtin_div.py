import ir
import pytest


def test_ge0_floor_div_ge0():
    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        with ir.Assert(a[()] >= 0):
            with ir.Assert(b[()] >= 0):
                c[()] = a[()] // b[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        with ir.Assert(a[()] >= 0):
            with ir.Assert(b[()] >= 0):
                c[()] = ir.round_towards_0_div(a[()], b[()])
    std = ir.pop_ast()

    assert std.match(ast)


def test_unknown_floor_div_unknown():
    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        c[()] = a[()] // b[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        c[()] = a[()] // b[()]
    std = ir.pop_ast()

    assert std.match(ast)


def test_ge0_ceil_div_ge0():
    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        with ir.Assert(a[()] >= 0):
            with ir.Assert(b[()] >= 0):
                c[()] = ir.ceil_div(a[()], b[()])
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        with ir.Assert(a[()] >= 0):
            with ir.Assert(b[()] >= 0):
                c[()] = ir.round_towards_0_div(a[()] + (b[()] - 1), b[()])
    std = ir.pop_ast()

    assert std.match(ast)


def test_unknown_ceil_div_unknown():
    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        c[()] = ir.ceil_div(a[()], b[()])
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        c[()] = ir.ceil_div(a[()], b[()])
    std = ir.pop_ast()

    assert std.match(ast)


def test_ge0_mod_ge0():
    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        with ir.Assert(a[()] >= 0):
            with ir.Assert(b[()] >= 0):
                c[()] = a[()] % b[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        with ir.Assert(a[()] >= 0):
            with ir.Assert(b[()] >= 0):
                c[()] = ir.remainder(a[()], b[()])
    std = ir.pop_ast()

    assert std.match(ast)


def test_mod_mod_ge0():
    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "input", "cpu"),
        ("d", (), "int32", "output", "cpu"),
    ]) as (a, b, c, d):
        with ir.Assert(a[()] >= 0):
            with ir.Assert(b[()] >= 0):
                with ir.Assert(c[()] >= 0):
                    d[()] = (a[()] % b[()]) % c[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "input", "cpu"),
        ("d", (), "int32", "output", "cpu"),
    ]) as (a, b, c, d):
        with ir.Assert(a[()] >= 0):
            with ir.Assert(b[()] >= 0):
                with ir.Assert(c[()] >= 0):
                    d[()] = ir.remainder(ir.remainder(a[()], b[()]), c[()])
    std = ir.pop_ast()

    assert std.match(ast)


def test_unknown_mod_unknown():
    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        c[()] = a[()] % b[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("c", (), "int32", "output", "cpu"),
    ]) as (a, b, c):
        c[()] = a[()] % b[()]
    std = ir.pop_ast()

    assert std.match(ast)
