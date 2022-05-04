import freetensor as ft
import pytest

# For normal cases, see test/codegen


def test_unsolvable_dependency():
    with ft.VarDef("y", (5,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4, nid="L1") as i:
            with ft.For("j", i, i + 2, nid="L2") as j:
                y[j] = i
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.parallelize("L1", "openmp")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_sharing_locals():
    with ft.VarDef([("x", (100,), "int32", "input", "gpu/global"),
                    ("t", (100,), "int32", "cache", "gpu/local"),
                    ("y", (100,), "int32", "output", "gpu/global")]) as (x, t,
                                                                         y):
        with ft.For("i", 0, 100, nid="L1") as i:
            t[i] = x[i] * 2
        with ft.For("i", 0, 100, nid="L2") as i:
            y[i] = t[i] + 1
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.parallelize("L1", "threadIdx.x")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_not_found():
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = i
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.parallelize("L1", "openmp")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_nested_thread_idx():
    with ft.VarDef("y", (4, 4), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4, nid='L1') as i:
            with ft.For("j", 0, 4, nid='L2') as j:
                y[i, j] = i + j
    ast = ft.pop_ast()
    print(ast)
    s = ft.Schedule(ast)
    s.parallelize("L1", "threadIdx.x")
    ast = s.ast()
    with pytest.raises(ft.InvalidSchedule):
        s.parallelize("L2", "threadIdx.x")
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


def test_no_deps():

    @ft.transform
    def test(ptr, edge1, edge2):
        ptr: ft.Var((11,), "int32", "input", "cpu")
        edge1: ft.Var((50,), "int32", "input", "cpu")
        edge2: ft.Var((50,), "int32", "output", "cpu")
        'nid: Li'
        'no_deps: edge2'
        for i in range(10):
            for j in range(ptr[i], ptr[i + 1]):
                edge2[j] = edge1[j] + i

    print(test)
    s = ft.Schedule(test)
    s.parallelize("Li", "openmp")  # No exception here
    print(s.ast())
