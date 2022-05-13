import freetensor as ft
import pytest


def test_basic():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(1000, 1000), "float32", "input", "cpu"]
        b: ft.Var[(1000, 1000), "float32", "input", "cpu"]
        c: ft.Var[(1000, 1000), "float32", "output", "cpu"]
        #! nid: Li
        for i in range(1000):
            for j in range(1000):
                c[i, j] = 0
                for k in range(1000):
                    c[i, j] += a[i, k] * b[k, j]

    print(test)
    s = ft.Schedule(test)
    s.auto_use_lib(ft.CPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == ["as_matmul(Li)"]


def test_fission_when_prefer_libs():

    @ft.transform
    def test(a, b1, b2, c1, c2):
        a: ft.Var[(1000, 1000), "float32", "input", "cpu"]
        b1: ft.Var[(1000, 1000), "float32", "input", "cpu"]
        b2: ft.Var[(1000, 1000), "float32", "input", "cpu"]
        c1: ft.Var[(1000, 1000), "float32", "inout", "cpu"]
        c2: ft.Var[(1000, 1000), "float32", "inout", "cpu"]
        #! nid: Li
        #! prefer_libs
        for i in range(1000):
            for j in range(1000):
                for k in range(1000):
                    c1[i, j] += a[i, k] * b1[k, j]
                    c2[i, j] += a[i, k] * b2[k, j]

    print(test)
    s = ft.Schedule(test)
    s.auto_use_lib(ft.CPU())
    print(s.ast())
    print(s.logs())
    assert "as_matmul(Li.0.lib)" in s.logs()
    assert "as_matmul(Li.1.lib)" in s.logs()
