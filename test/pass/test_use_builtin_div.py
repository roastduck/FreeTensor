import ir
import pytest


def test_ge0_floor_div_ge0():
    with ir.VarDef(
        [
            ("a", (), "int32", "input", "cpu"),
            ("b", (), "int32", "input", "cpu"),
            ("c", (), "int32", "output", "cpu"),
        ]
    ) as (a, b, c):
        with ir.Assert(a[()] >= 0):
            with ir.Assert(b[()] >= 0):
                c[()] = a[()] // b[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef(
        [
            ("a", (), "int32", "input", "cpu"),
            ("b", (), "int32", "input", "cpu"),
            ("c", (), "int32", "output", "cpu"),
        ]
    ) as (a, b, c):
        with ir.Assert(a[()] >= 0):
            with ir.Assert(b[()] >= 0):
                c[()] = ir.round_towards_0_div(a[()], b[()])
    std = ir.pop_ast()

    assert std.match(ast)


def test_unknown_floor_div_unknown():
    with ir.VarDef(
        [
            ("a", (), "int32", "input", "cpu"),
            ("b", (), "int32", "input", "cpu"),
            ("c", (), "int32", "output", "cpu"),
        ]
    ) as (a, b, c):
        c[()] = a[()] // b[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef(
        [
            ("a", (), "int32", "input", "cpu"),
            ("b", (), "int32", "input", "cpu"),
            ("c", (), "int32", "output", "cpu"),
        ]
    ) as (a, b, c):
        c[()] = a[()] // b[()]
    std = ir.pop_ast()

    assert std.match(ast)


def test_ge0_ceil_div_ge0():
    with ir.VarDef(
        [
            ("a", (), "int32", "input", "cpu"),
            ("b", (), "int32", "input", "cpu"),
            ("c", (), "int32", "output", "cpu"),
        ]
    ) as (a, b, c):
        with ir.Assert(a[()] >= 0):
            with ir.Assert(b[()] >= 0):
                c[()] = ir.ceil_div(a[()], b[()])
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef(
        [
            ("a", (), "int32", "input", "cpu"),
            ("b", (), "int32", "input", "cpu"),
            ("c", (), "int32", "output", "cpu"),
        ]
    ) as (a, b, c):
        with ir.Assert(a[()] >= 0):
            with ir.Assert(b[()] >= 0):
                c[()] = ir.round_towards_0_div(a[()] + (b[()] - 1), b[()])
    std = ir.pop_ast()

    assert std.match(ast)


def test_unknown_ceil_div_unknown():
    with ir.VarDef(
        [
            ("a", (), "int32", "input", "cpu"),
            ("b", (), "int32", "input", "cpu"),
            ("c", (), "int32", "output", "cpu"),
        ]
    ) as (a, b, c):
        c[()] = ir.ceil_div(a[()], b[()])
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef(
        [
            ("a", (), "int32", "input", "cpu"),
            ("b", (), "int32", "input", "cpu"),
            ("c", (), "int32", "output", "cpu"),
        ]
    ) as (a, b, c):
        c[()] = ir.ceil_div(a[()], b[()])
    std = ir.pop_ast()

    assert std.match(ast)
