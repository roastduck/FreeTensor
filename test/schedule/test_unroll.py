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
    ast = ir.pop_ast()

    s = ir.Schedule(ast)
    code, params = ir.codegen(s.ast(), target)
    with pytest.raises(ir.InvalidSchedule):
        s.unroll("L0")
    code_, params_ = ir.codegen(s.ast(), target)

    assert code == code_


def test_not_constant():
    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, n[()], nid="L1") as i:
            y[i] = i

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    code, params = ir.codegen(s.ast(), target)
    with pytest.raises(ir.InvalidSchedule):
        s.unroll("L1")
    code_, params_ = ir.codegen(s.ast(), target)

    assert code == code_


def test_unbounded_length():
    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("x", (4, 4), "int32", "output", "cpu")]) as (n, x):
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", i, 4, nid="L2") as j:
                x[i, j] = 1

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    code, params = ir.codegen(s.ast(), target)
    with pytest.raises(ir.InvalidSchedule):
        s.unroll("L2")
    code_, params_ = ir.codegen(s.ast(), target)

    assert code == code_


def test_constant_length():
    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("x", (8,), "int32", "output", "cpu")]) as (n, x):
        with ir.For("i", n[()], n[()] + 4, nid="L1") as i:
            x[i] = 1

    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    code, params = ir.codegen(s.ast(), target)
    s.unroll("L1")
    code_, params_ = ir.codegen(s.ast(), target)

    assert code != code_
