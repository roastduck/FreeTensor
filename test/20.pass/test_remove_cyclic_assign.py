import freetensor as ft
import pytest


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_basic(as_subprocess):
    with ft.VarDef([("a", (), "int32", "inout", "cpu"),
                    ("b", (), "int32", "output", "cpu")]) as (a, b):
        b[()] = a[()]
        a[()] = b[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast, verbose=1, as_subprocess=as_subprocess)

    with ft.VarDef([("a", (), "int32", "inout", "cpu"),
                    ("b", (), "int32", "output", "cpu")]) as (a, b):
        b[()] = a[()]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_in_branch(as_subprocess):
    with ft.VarDef([("cond", (), "bool", "input", "cpu"),
                    ("a", (), "int32", "inout", "cpu"),
                    ("b", (), "int32", "output", "cpu")]) as (cond, a, b):
        with ft.If(cond[()]):
            b[()] = a[()]
            a[()] = b[()]
    ast = ft.pop_ast(verbose=True)
    # Run this pass only. No sinking vars into If
    ast = ft.remove_cyclic_assign(ast, as_subprocess=as_subprocess)
    print(ast)

    with ft.VarDef([("cond", (), "bool", "input", "cpu"),
                    ("a", (), "int32", "inout", "cpu"),
                    ("b", (), "int32", "output", "cpu")]) as (cond, a, b):
        with ft.If(cond[()]):
            b[()] = a[()]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_in_branch_2(as_subprocess):
    with ft.VarDef([("p", (4,), "int32", "input", "cpu"),
                    ("a", (4,), "int32", "inout", "cpu"),
                    ("b", (4,), "int32", "output", "cpu")]) as (p, a, b):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("cond", (), "bool", "cache", "cpu") as cond:
                cond[...] = p[i] > 0
                with ft.If(cond[...]):
                    b[i] = a[i]
                    a[i] = b[i]
    ast = ft.pop_ast(verbose=True)
    # Run this pass only. No sinking vars into If
    ast = ft.remove_cyclic_assign(ast, as_subprocess=as_subprocess)
    print(ast)

    with ft.VarDef([("p", (4,), "int32", "input", "cpu"),
                    ("a", (4,), "int32", "inout", "cpu"),
                    ("b", (4,), "int32", "output", "cpu")]) as (p, a, b):
        with ft.For("i", 0, 4) as i:
            with ft.VarDef("cond", (), "bool", "cache", "cpu") as cond:
                cond[...] = p[i] > 0
                with ft.If(cond[...]):
                    b[i] = a[i]
    std = ft.pop_ast()

    assert std.match(ast)
