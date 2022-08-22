import freetensor as ft
import pytest

device = ft.CPU()
target = device.target()

# Please refer to test/codegen for some architecture dependent test cases


def test_not_found():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            y[i] = x[i] + 1
    func = ft.Func("main", ["x", "y"], [], ft.pop_ast())

    s = ft.Schedule(func)
    code = ft.codegen(s.func(), target)
    with pytest.raises(ft.InvalidSchedule):
        s.unroll("L0")
    code_ = ft.codegen(s.func(), target)

    assert str(code) == str(code_)


def test_not_constant():
    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ft.For("i", 0, n[()], label="L1") as i:
            y[i] = i

    func = ft.Func("main", ["n", "y"], [], ft.pop_ast())
    print(func)
    s = ft.Schedule(func)
    code = ft.codegen(s.func(), target)
    with pytest.raises(ft.InvalidSchedule):
        s.unroll("L1")
    code_ = ft.codegen(s.func(), target)

    assert str(code) == str(code_)


def test_unbounded_length():
    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("x", (4, 4), "int32", "output", "cpu")]) as (n, x):
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", i, 4, label="L2") as j:
                x[i, j] = 1

    func = ft.Func("main", ["n", "x"], [], ft.pop_ast())
    print(func)
    s = ft.Schedule(func)
    code = ft.codegen(s.func(), target)
    with pytest.raises(ft.InvalidSchedule):
        s.unroll("L2")
    code_ = ft.codegen(s.func(), target)

    assert str(code) == str(code_)


def test_constant_length():
    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("x", (8,), "int32", "output", "cpu")]) as (n, x):
        with ft.For("i", n[()], n[()] + 4, label="L1") as i:
            x[i] = 1

    func = ft.Func("main", ["n", "x"], [], ft.pop_ast())
    print(func)
    s = ft.Schedule(func)
    code = ft.codegen(s.func(), target)
    s.unroll("L1")
    code_ = ft.codegen(s.func(), target)

    assert code != code_


def test_immediate_basic():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            y[i] = x[i] + 1

    s = ft.Schedule(ft.pop_ast())
    s.unroll("L1", True)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        y[0] = x[0] + 1
        y[1] = x[1] + 1
        y[2] = x[2] + 1
        y[3] = x[3] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_immediate_with_offset():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        y[0] = 0
        with ft.For("i", 1, 4, label="L1") as i:
            y[i] = x[i] + 1

    s = ft.Schedule(ft.pop_ast())
    s.unroll("L1", True)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        y[0] = 0
        y[1] = x[1] + 1
        y[2] = x[2] + 1
        y[3] = x[3] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_immediate_with_step():
    with ft.VarDef([("x", (8,), "int32", "input", "cpu"),
                    ("y", (8,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 6, -2, -2, label="L1") as i:
            y[i] = x[i] + 1

    s = ft.Schedule(ft.pop_ast())
    s.unroll("L1", True)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (8,), "int32", "input", "cpu"),
                    ("y", (8,), "int32", "output", "cpu")]) as (x, y):
        y[6] = x[6] + 1
        y[4] = x[4] + 1
        y[2] = x[2] + 1
        y[0] = x[0] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_folding_sum():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 0
        with ft.For("i", 0, 4, label="L1") as i:
            y[()] = y[()] + x[i]

    s = ft.Schedule(ft.pop_ast())
    s.unroll("L1", True)
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = ft.any()  # Sum in any association
    std = ft.pop_ast()

    assert std.match(ast)
