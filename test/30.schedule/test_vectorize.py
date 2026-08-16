import freetensor as ft
import pytest

device = ft.CPU()
target = device.target()

# This file only tests for invalid cases. Valid cases are tested per backend
# in test/40.codegen


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_not_found(as_subprocess):
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4, label="L1") as i:
            y[i] = x[i] + 1
    func = ft.Func("main", ["x", "y"], [], ft.pop_ast())

    s = ft.Schedule(func)
    code = ft.codegen(s.func(), target)
    with pytest.raises(ft.InvalidSchedule):
        s.vectorize("L0", as_subprocess=as_subprocess)
    code_ = ft.codegen(s.func(), target)

    assert str(code) == str(code_)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_dep_not_met(as_subprocess):
    with ft.VarDef("y", (5,), "int32", "inout", "cpu") as y:
        with ft.For("i", 1, 5, label="L1") as i:
            y[i] = y[i - 1] + y[i]
    func = ft.Func("main", ["y"], [], ft.pop_ast())

    s = ft.Schedule(func)
    code = ft.codegen(s.func(), target)
    with pytest.raises(ft.InvalidSchedule):
        s.vectorize("L1", as_subprocess=as_subprocess)
    code_ = ft.codegen(s.func(), target)

    assert str(code) == str(code_)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_dep_not_met_reduction(as_subprocess):
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[...] = 0
        with ft.For("i", 0, 4, label="L1") as i:
            y[...] += x[i]
    func = ft.Func("main", ["x", "y"], [], ft.pop_ast())

    s = ft.Schedule(func)
    code = ft.codegen(s.func(), target)
    with pytest.raises(ft.InvalidSchedule):
        s.vectorize("L1", as_subprocess=as_subprocess)
    code_ = ft.codegen(s.func(), target)

    assert str(code) == str(code_)
