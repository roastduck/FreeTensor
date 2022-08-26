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
    print(s.pretty_logs())
    assert s.pretty_logs() == ["fuse(L1, L2, true)"]


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
    print(s.pretty_logs())
    assert s.pretty_logs() == ["fuse(L1, L3, true)", "fuse(L2, L4, true)"]


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
    print(s.pretty_logs())
    assert s.pretty_logs() == ["move_to(L1, before, L2)", "fuse(L1, L2, true)"]


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
    print(s.pretty_logs())
    assert s.pretty_logs() == ["move_to(L2, after, L1)", "fuse(L1, L2, true)"]


def test_tune_fuse():
    # We may fuse these loops. But fusing them will make it impossible to parallelize.
    # After tuning, we will end up in not fusing them
    with ft.VarDef([("a", (100, 100, 100), "int32", "input", "cpu"),
                    ("b", (100, 100, 100), "int32", "output", "cpu"),
                    ("c", (100, 100, 100), "int32", "output", "cpu")]) as (a, b,
                                                                           c):
        with ft.For("i", 0, 100, label="Li1") as i:
            with ft.For("j", 0, 100, label="Lj1") as j:
                with ft.For("k", 0, 100, label="Lk1") as k:
                    b[i, j, k] = ft.if_then_else(
                        j > 0 and k > 0, b[i, j - 1, k - 1], 0) + a[i, j, k]
        with ft.For("i", 0, 100, label="Li2") as i:
            with ft.For("j", 0, 100, label="Lj2") as j:
                with ft.For("k", 0, 100, label="Lk2") as k:
                    c[i, j, k] = ft.if_then_else(i > 0, c[i - 1, j, k - 1],
                                                 0) + a[i, j, k]

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
    print(s.pretty_logs())

    for log in s.pretty_logs():
        assert "fuse" not in log


def test_tune_fission():
    # The reverse schedule of `test_tune_fuse`
    with ft.VarDef([("a", (100, 100, 100), "int32", "input", "cpu"),
                    ("b", (100, 100, 100), "int32", "output", "cpu"),
                    ("c", (100, 100, 100), "int32", "output", "cpu")]) as (a, b,
                                                                           c):
        with ft.For("i", 0, 100, label="Li") as i:
            with ft.For("j", 0, 100, label="Lj") as j:
                with ft.For("k", 0, 100, label="Lk") as k:
                    b[i, j, k] = ft.if_then_else(
                        j > 0 and k > 0, b[i, j - 1, k - 1], 0) + a[i, j, k]
                    c[i, j, k] = ft.if_then_else(i > 0, c[i - 1, j, k - 1],
                                                 0) + a[i, j, k]

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
    print(s.pretty_logs())

    assert "fission" in ", ".join(s.pretty_logs())


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_tune_with_cond():
    # Fuse loops that can parallelize. Don't fuse loops that can't
    with ft.VarDef([("a", (100, 100, 100), "int32", "input", "gpu/global"),
                    ("b", (100, 100, 100), "int32", "output", "gpu/global"),
                    ("c", (100, 100, 100), "int32", "output", "gpu/global"),
                    ("y", (100, 100, 100), "int32", "output", "gpu/global")
                   ]) as (a, b, c, y):
        # Fusing L1 and L2 leads to poor parallelization, which is not preferred
        with ft.For("i", 0, 100, label="Li1") as i:
            with ft.For("j", 0, 100, label="Lj1") as j:
                with ft.For("k", 0, 100, label="Lk1") as k:
                    b[i, j, k] = ft.if_then_else(
                        j > 0 and k > 0, b[i, j - 1, k - 1], 0) + a[i, j, k]
        with ft.For("i", 0, 100, label="Li2") as i:
            with ft.For("j", 0, 100, label="Lj2") as j:
                with ft.For("k", 0, 100, label="Lk2") as k:
                    c[i, j, k] = ft.if_then_else(i > 0, c[i - 1, j, k - 1],
                                                 0) + a[i, j, k]
        # Fusing L3 and L4 is favorable to reduce kernel launch count
        with ft.VarDef("t", (100, 100, 100), "int32", "cache",
                       "gpu/global") as t:
            with ft.For("i", 0, 100, label="Li3") as i:
                with ft.For("j", 0, 100, label="Lj3") as j:
                    with ft.For("k", 0, 100, label="Lk3") as k:
                        t[i, j, k] = a[i, j, k] * i
            with ft.For("i", 0, 100, label="Li4") as i:
                with ft.For("j", 0, 100, label="Lj4") as j:
                    with ft.For("k", 0, 100, label="Lk4") as k:
                        y[i, j, k] = t[i, j, k] * t[i, j, k]

    func = ft.Func("main", ["a", "b", "c", "y"], [], ft.pop_ast(verbose=True))
    s = ft.Schedule(func)
    a = ft.Array(np.random.randint(0, 100, (100, 100, 100)).astype("int32"))
    b = ft.Array(np.zeros((100, 100, 100), dtype="int32"))
    c = ft.Array(np.zeros((100, 100, 100), dtype="int32"))
    y = ft.Array(np.zeros((100, 100, 100), dtype="int32"))
    trials = s.tune_auto_schedule(10,
                                  1,
                                  ft.GPU(), (a, b, c, y),
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
    print(s.pretty_logs())

    assert "fuse(Li1, Li2, true)" not in s.pretty_logs()
    assert "fuse(Li3, Li4, true)" in s.pretty_logs()
