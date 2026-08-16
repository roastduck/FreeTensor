import freetensor as ft
import pytest

# For normal cases, see test/codegen


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_unsolvable_dependence(as_subprocess):
    with ft.VarDef("y", (5,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4, label="L1") as i:
            with ft.For("j", i, i + 2, label="L2") as j:
                y[j] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.parallelize("L1", "openmp", as_subprocess=as_subprocess)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_sharing_locals(as_subprocess):
    with ft.VarDef([("x", (100,), "int32", "input", "gpu/global"),
                    ("t", (100,), "int32", "cache", "gpu/local"),
                    ("y", (100,), "int32", "output", "gpu/global")]) as (x, t,
                                                                         y):
        with ft.For("i", 0, 100, label="L1") as i:
            t[i] = x[i] * 2
        with ft.For("i", 0, 100, label="L2") as i:
            y[i] = t[i] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.parallelize("L1", "threadIdx.x", as_subprocess=as_subprocess)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_not_found(as_subprocess):
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    with pytest.raises(ft.InvalidSchedule):
        s.parallelize("L1", "openmp", as_subprocess=as_subprocess)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_nested_thread_idx_1(as_subprocess):
    with ft.VarDef("y", (4, 4), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4, label='L1') as i:
            with ft.For("j", 0, 4, label='L2') as j:
                y[i, j] = i
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.parallelize("L1", "threadIdx.x", as_subprocess=as_subprocess)
    ast = s.ast()
    with pytest.raises(ft.InvalidSchedule):
        s.parallelize("L2", "threadIdx.x", as_subprocess=as_subprocess)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_nested_thread_idx_2(as_subprocess):
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4, label='L1') as i:
            with ft.VarDef("t", (), "int32", "cache", "gpu/global") as t:
                t[...] = i
                with ft.For("j", 0, 4, label='L2') as j:
                    y[j] += t[...] + j
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.parallelize("L1", "threadIdx.x", as_subprocess=as_subprocess)
    ast = s.ast()
    with pytest.raises(ft.InvalidSchedule):
        s.parallelize("L2", "threadIdx.x", as_subprocess=as_subprocess)
    ast_ = s.ast()  # Should not changed
    assert ast_.match(ast)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_no_deps(as_subprocess):

    @ft.transform
    def test(ptr, edge1, edge2):
        ptr: ft.Var[(11,), "int32", "input", "cpu"]
        edge1: ft.Var[(50,), "int32", "input", "cpu"]
        edge2: ft.Var[(50,), "int32", "output", "cpu"]
        #! label: Li
        #! no_deps: edge2
        for i in range(10):
            for j in range(ptr[i], ptr[i + 1]):
                edge2[j] = edge1[j] + i

    print(test)
    s = ft.Schedule(test)
    s.parallelize("Li", "openmp",
                  as_subprocess=as_subprocess)  # No exception here
    print(s.ast())
