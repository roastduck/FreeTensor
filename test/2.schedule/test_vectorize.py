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
    func = ir.Func("main", ["x", "y"], ir.pop_ast())

    s = ir.Schedule(func)
    code = ir.codegen(s.func(), target)
    with pytest.raises(ir.InvalidSchedule):
        s.vectorize("L0")
    code_ = ir.codegen(s.func(), target)

    assert code == code_


def test_dep_not_met():
    with ir.VarDef("y", (5,), "int32", "inout", "cpu") as y:
        with ir.For("i", 1, 5, nid="L1") as i:
            y[i] = y[i - 1] + y[i]
    func = ir.Func("main", ["y"], ir.pop_ast())

    s = ir.Schedule(func)
    code = ir.codegen(s.func(), target)
    with pytest.raises(ir.InvalidSchedule):
        s.vectorize("L1")
    code_ = ir.codegen(s.func(), target)

    assert code == code_
