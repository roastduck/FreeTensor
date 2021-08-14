import ir
import pytest

# For normal cases, see test/codegen


def test_unsolvable_dependency():
    with ir.VarDef("y", (5,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4, nid="L1") as i:
            with ir.For("j", i, i + 2, nid="L2") as j:
                y[j] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.parallelize("L1", "openmp")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_not_found():
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = i
    ast = ir.pop_ast()
    print(ast)
    s = ir.Schedule(ast)
    with pytest.raises(ir.InvalidSchedule):
        s.parallelize("L1", "openmp")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)
