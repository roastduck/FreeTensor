import ir
import pytest

target = ir.CPU()
device = ir.Device(target)

# For normal test cases, please refer to test/codegen


def test_not_found():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4, nid="L1") as i:
            y[i] = x[i] + 1
    func = ir.Func(["x", "y"], ir.pop_ast())

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

    func = ir.Func(["n", "y"], ir.pop_ast())
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

    func = ir.Func(["n", "x"], ir.pop_ast())
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

    func = ir.Func(["n", "x"], ir.pop_ast())
    print(func)
    s = ir.Schedule(func)
    code = ir.codegen(s.func(), target)
    s.unroll("L1")
    code_ = ir.codegen(s.func(), target)

    assert code != code_
