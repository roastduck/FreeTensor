import freetensor as ft
import pytest
import numpy as np


def test_basic():
    with ft.VarDef([("a", (1000,), "int32", "input", "cpu"),
                    ("b", (1000,), "int32", "input", "cpu"),
                    ("c", (1000,), "int32", "input", "cpu"),
                    ("y", (1000,), "int32", "output", "cpu")]) as (a, b, c, y):
        with ft.For("i", 0, 1000, label="L1") as i:
            y[i] = a[i] + b[i]
        with ft.For("i", 0, 1000, label="L2") as i:
            y[i] += c[i]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_fission_fuse(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert logs == ["fuse(L1, L2, true)"]


def test_nested():
    with ft.VarDef([("a", (10, 10), "int32", "input", "cpu"),
                    ("b", (10, 10), "int32", "input", "cpu"),
                    ("c", (10, 10), "int32", "input", "cpu"),
                    ("y", (10, 10), "int32", "output", "cpu")]) as (a, b, c, y):
        with ft.For("i", 0, 10, label="L1") as i:
            with ft.For("j", 0, 10, label="L2") as j:
                y[i, j] = a[i, j] + b[i, j]
        with ft.For("i", 0, 10, label="L3") as i:
            with ft.For("j", 0, 10, label="L4") as j:
                y[i, j] += c[i, j]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_fission_fuse(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert logs == ["fuse(L1, L3, true)", "fuse(L2, L4, true)"]


def test_stmt_in_between_1():
    with ft.VarDef([("x1", (1000,), "int32", "input", "cpu"),
                    ("x2", (1000,), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x1, x2, y1, y2):
        ft.MarkLabel('S1')
        y1[()] = 0
        with ft.For("i", 0, 1000, label="L1") as i:
            y1[()] += x1[i]
        ft.MarkLabel('S2')
        y2[()] = 0
        with ft.For("i", 0, 1000, label="L2") as i:
            y2[()] += x2[i]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_fission_fuse(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert logs == ["swap(S2, L1)", "fuse(L1, L2, true)"]


def test_stmt_in_between_2():
    with ft.VarDef([("x1", (), "int32", "inout", "cpu"),
                    ("x2", (), "int32", "inout", "cpu"),
                    ("y1", (1000,), "int32", "output", "cpu"),
                    ("y2", (1000,), "int32", "output", "cpu")]) as (x1, x2, y1,
                                                                    y2):
        with ft.For("i", 0, 1000, label="L1") as i:
            y1[i] = x1[()] * i
        ft.MarkLabel('S1')
        x1[()] = 0
        with ft.For("i", 0, 1000, label="L2") as i:
            y2[i] = x2[()] * i
        ft.MarkLabel('S2')
        x2[()] = 0

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_fission_fuse(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert logs == ["swap(L2, S1)", "fuse(L1, L2, true)"]


def test_tune_fuse():
    # Plan 1: Fuse these loops, which makes it impossible to parallelize
    # Plan 2: Not fusing these loops, then we can parallelize them
    # We should decide on real measurement. (Parallelization does not always bring speedup,
    # especially when there are too many cores).
    with ft.VarDef([("a", (100, 100, 100), "int32", "input", "cpu"),
                    ("b", (100, 100, 100), "int32", "inout", "cpu"),
                    ("c", (100, 100, 100), "int32", "inout", "cpu")]) as (a, b,
                                                                          c):
        with ft.For("i", 0, 100, label="Li1") as i:
            with ft.For("j", 0, 100, label="Lj1") as j:
                with ft.For("k", 0, 100, label="Lk1") as k:
                    b[i, j,
                      k] = b[i,
                             (j + 1) % 100, k] + b[i, j,
                                                   (k + 1) % 100] + a[i, j, k]
        with ft.For("i", 0, 100, label="Li2") as i:
            with ft.For("j", 0, 100, label="Lj2") as j:
                with ft.For("k", 0, 100, label="Lk2") as k:
                    c[i, j, k] = c[(i + 1) % 100, j,
                                   k] + c[i, j, (k + 1) % 100] + a[i, j, k]

    func = ft.Func("main", ["a", "b", "c"], [], ft.pop_ast(verbose=True))
    s = ft.Schedule(func)
    a = ft.Array(np.random.randint(0, 100, (100, 100, 100)).astype("int32"))
    b = ft.Array(np.zeros((100, 100, 100), dtype="int32"))
    c = ft.Array(np.zeros((100, 100, 100), dtype="int32"))
    trials = s.tune_auto_schedule(10, 1, ft.CPU(), (a, b, c), to_learn="fuse")
    traces = [
        "{}:\n t={}, stddev={}".format(
            "\n".join([str(obs)
                       for obs in trial.trace]), trial.time, trial.stddev)
        for trial in trials
    ]
    print("\n-------\n".join(traces))

    s.auto_schedule(ft.CPU())
    print(s.func())
    logs = list(map(str, s.logs()))
    print(logs)

    s_plan1 = ft.Schedule(func)
    s_plan1.fuse("Li1", "Li2")
    s_plan1.fuse("Lj1", "Lj2")
    s_plan1.fuse("Lk1", "Lk2")

    s_plan2 = ft.Schedule(func)
    s_plan2.parallelize("Li1", "openmp")
    s_plan2.reorder(["Lj2", "Li2"])
    s_plan2.parallelize("Lj2", "openmp")

    exe1 = ft.optimize(s_plan1.func())
    exe2 = ft.optimize(s_plan2.func())
    for i in range(3):
        t1, stddev1 = exe1.time(a, b, c)
        if stddev1 <= 0.1 * t1:
            break
        print("Rerunning because stddev is too high")
    for i in range(3):
        t2, stddev2 = exe2.time(a, b, c)
        if stddev2 <= 0.1 * t2:
            break
        print("Rerunning because stddev is too high")
    print(f"t1 = {t1}ms, stddev1 = {stddev1}ms")
    print(f"t2 = {t2}ms, stddev2 = {stddev2}ms")
    if (t1 < t2):
        assert "fuse" in ", ".join(logs)
    else:
        assert "fuse" not in ", ".join(logs)


def test_tune_fission():
    # The reverse schedule of `test_tune_fuse`

    with ft.VarDef([("a", (100, 100, 100), "int32", "input", "cpu"),
                    ("b", (100, 100, 100), "int32", "inout", "cpu"),
                    ("c", (100, 100, 100), "int32", "inout", "cpu")]) as (a, b,
                                                                          c):
        with ft.For("i", 0, 100, label="Li") as i:
            with ft.For("j", 0, 100, label="Lj") as j:
                with ft.For("k", 0, 100, label="Lk") as k:
                    ft.MarkLabel("S0")
                    b[i, j,
                      k] = b[i,
                             (j + 1) % 100, k] + b[i, j,
                                                   (k + 1) % 100] + a[i, j, k]
                    c[i, j, k] = c[(i + 1) % 100, j,
                                   k] + c[i, j, (k + 1) % 100] + a[i, j, k]

    func = ft.Func("main", ["a", "b", "c"], [], ft.pop_ast(verbose=True))
    s = ft.Schedule(func)
    a = ft.Array(np.random.randint(0, 100, (100, 100, 100)).astype("int32"))
    b = ft.Array(np.zeros((100, 100, 100), dtype="int32"))
    c = ft.Array(np.zeros((100, 100, 100), dtype="int32"))
    trials = s.tune_auto_schedule(10, 1, ft.CPU(), (a, b, c), to_learn="fuse")
    traces = [
        "{}:\n t={}, stddev={}".format(
            "\n".join([str(obs)
                       for obs in trial.trace]), trial.time, trial.stddev)
        for trial in trials
    ]
    print("\n-------\n".join(traces))

    s.auto_schedule(ft.CPU())
    print(s.func())
    logs = list(map(str, s.logs()))
    print(logs)

    s_plan1 = ft.Schedule(func)
    s_plan1.fission("Lk", ft.FissionSide.After, "S0")
    s_plan1.fission("Lj", ft.FissionSide.After, "$fission.0{S0}")
    s_plan1.fission("Li", ft.FissionSide.After, "$fission.0{$fission.0{S0}}")
    s_plan1.parallelize("$fission.0{Li}", "openmp")
    s_plan1.reorder(["$fission.1{$fission.1{Lj}}", "$fission.1{Li}"])
    s_plan1.parallelize("$fission.1{$fission.1{Lj}}", "openmp")

    s_plan2 = ft.Schedule(func)
    # Do nothing

    exe1 = ft.optimize(s_plan1.func())
    exe2 = ft.optimize(s_plan2.func())
    for i in range(3):
        t1, stddev1 = exe1.time(a, b, c)
        if stddev1 <= 0.1 * t1:
            break
        print("Rerunning because stddev is too high")
    for i in range(3):
        t2, stddev2 = exe2.time(a, b, c)
        if stddev2 <= 0.1 * t2:
            break
        print("Rerunning because stddev is too high")
    print(f"t1 = {t1}ms, stddev1 = {stddev1}ms")
    print(f"t2 = {t2}ms, stddev2 = {stddev2}ms")
    if (t1 < t2):
        assert "fission" in ", ".join(logs)
    else:
        assert "fission" not in ", ".join(logs)


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_tune_with_cond():
    # Test different dicisions in a single program
    # Fuse loops that can parallelize. Don't fuse loops that can't
    with ft.VarDef([("a", (100, 100, 10), "int32", "input", "gpu/global"),
                    ("b", (100, 100, 10), "int32", "inout", "gpu/global"),
                    ("c", (100, 100, 10), "int32", "inout", "gpu/global"),
                    ("y", (100, 100, 10), "int32", "output", "gpu/global"),
                    ("z", (100, 100, 10), "int32", "output", "gpu/global")
                   ]) as (a, b, c, y, z):
        # Fusing L1 and L2 leads to poor parallelization, which is not preferred
        with ft.For("i", 0, 100, label="Li1") as i:
            with ft.For("j", 0, 100, label="Lj1") as j:
                with ft.For("k", 0, 10, label="Lk1") as k:
                    b[i, j,
                      k] = b[i, (j + 1) % 100, k] + b[i, j,
                                                      (k + 1) % 10] + a[i, j, k]
        with ft.For("i", 0, 100, label="Li2") as i:
            with ft.For("j", 0, 100, label="Lj2") as j:
                with ft.For("k", 0, 10, label="Lk2") as k:
                    c[i, j,
                      k] = c[(i + 1) % 100, j, k] + c[i, j,
                                                      (k + 1) % 10] + a[i, j, k]
        # Fusing L3 and L4 is favorable to reduce kernel launch count
        with ft.For("i", 0, 100, label="Li3") as i:
            with ft.For("j", 0, 100, label="Lj3") as j:
                with ft.For("k", 0, 10, label="Lk3") as k:
                    y[i, j, k] = a[i, j, k] * i
        with ft.For("i", 0, 100, label="Li4") as i:
            with ft.For("j", 0, 100, label="Lj4") as j:
                with ft.For("k", 0, 10, label="Lk4") as k:
                    z[i, j, k] = y[i, j, k] * y[i, j, k]

    func = ft.Func("main", ["a", "b", "c", "y", "z"], [],
                   ft.pop_ast(verbose=True))
    s = ft.Schedule(func)
    a = ft.Array(np.random.randint(0, 100, (100, 100, 10)).astype("int32"))
    b = ft.Array(np.zeros((100, 100, 10), dtype="int32"))
    c = ft.Array(np.zeros((100, 100, 10), dtype="int32"))
    y = ft.Array(np.zeros((100, 100, 10), dtype="int32"))
    z = ft.Array(np.zeros((100, 100, 10), dtype="int32"))
    trials = s.tune_auto_schedule(10,
                                  1,
                                  ft.GPU(), (a, b, c, y, z),
                                  to_learn="fuse")
    traces = [
        "{}:\n t={}, stddev={}".format(
            "\n".join([str(obs)
                       for obs in trial.trace]), trial.time, trial.stddev)
        for trial in trials
    ]
    print("\n-------\n".join(traces))

    s.auto_schedule(ft.GPU())
    print(s.func())
    logs = list(map(str, s.logs()))
    print(logs)

    assert "fuse(Li1, Li2, true)" not in logs
    assert "fuse(Li3, Li4, true)" in logs
