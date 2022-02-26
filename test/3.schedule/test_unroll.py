import ir
import pytest

target = ir.CPU()
device = ir.Device(target)

# Please refer to test/codegen for some architecture dependent test cases


def test_not_found():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            y[i] = x[i] + 1
    func = ir.Func("main", ["x", "y"], [], ir.pop_ast())

    s = ir.Schedule(func)
    code = ir.codegen(s.func(), target)
    with pytest.raises(ir.InvalidSchedule):
        s.unroll("L0")
    code_ = ir.codegen(s.func(), target)

    assert code == code_


def test_not_constant():
    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, n[()], nid="L1") as i:
            y[i] = i

    func = ir.Func("main", ["n", "y"], [], ir.pop_ast())
    print(func)
    s = ir.Schedule(func)
    code = ir.codegen(s.func(), target)
    with pytest.raises(ir.InvalidSchedule):
        s.unroll("L1")
    code_ = ir.codegen(s.func(), target)

    assert code == code_


def test_unbounded_length():
    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("x", (4, 4), "int32", "output", "cpu")]) as (n, x):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", i, 4, nid="L2") as j:
                x[i, j] = 1

    func = ir.Func("main", ["n", "x"], [], ir.pop_ast())
    print(func)
    s = ir.Schedule(func)
    code = ir.codegen(s.func(), target)
    with pytest.raises(ir.InvalidSchedule):
        s.unroll("L2")
    code_ = ir.codegen(s.func(), target)

    assert code == code_


def test_constant_length():
    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("x", (8,), "int32", "output", "cpu")]) as (n, x):
        with ir.For("i", n[()], n[()] + 4, nid="L1") as i:
            x[i] = 1

    func = ir.Func("main", ["n", "x"], [], ir.pop_ast())
    print(func)
    s = ir.Schedule(func)
    code = ir.codegen(s.func(), target)
    s.unroll("L1")
    code_ = ir.codegen(s.func(), target)

    assert code != code_


def test_immediate_basic():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            y[i] = x[i] + 1

    s = ir.Schedule(ir.pop_ast())
    s.unroll("L1", True)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        y[0] = x[0] + 1
        y[1] = x[1] + 1
        y[2] = x[2] + 1
        y[3] = x[3] + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_immediate_with_offset():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        y[0] = 0
        with ir.For("i", 1, 4, nid="L1") as i:
            y[i] = x[i] + 1

    s = ir.Schedule(ir.pop_ast())
    s.unroll("L1", True)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        y[0] = 0
        y[1] = x[1] + 1
        y[2] = x[2] + 1
        y[3] = x[3] + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_immediate_with_step():
    with ir.VarDef([("x", (8,), "int32", "input", "cpu"),
                    ("y", (8,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 6, -2, -2, nid="L1") as i:
            y[i] = x[i] + 1

    s = ir.Schedule(ir.pop_ast())
    s.unroll("L1", True)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (8,), "int32", "input", "cpu"),
                    ("y", (8,), "int32", "output", "cpu")]) as (x, y):
        y[6] = x[6] + 1
        y[4] = x[4] + 1
        y[2] = x[2] + 1
        y[0] = x[0] + 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_folding_sum():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 0
        with ir.For("i", 0, 4, nid="L1") as i:
            y[()] = y[()] + x[i]

    s = ir.Schedule(ir.pop_ast())
    s.unroll("L1", True)
    ast = s.ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = ir.any()  # Sum in any association
    std = ir.pop_ast()

    assert std.match(ast)
