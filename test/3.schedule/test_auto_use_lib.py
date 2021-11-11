import ir
import pytest


def test_basic():

    @ir.transform
    def test(a, b, c):
        ir.declare_var(a, (1000, 1000), "float32", "input", "cpu")
        ir.declare_var(b, (1000, 1000), "float32", "input", "cpu")
        ir.declare_var(c, (1000, 1000), "float32", "output", "cpu")
        'nid: Li'
        for i in range(1000):
            for j in range(1000):
                c[i, j] = 0
                for k in range(1000):
                    c[i, j] += a[i, k] * b[k, j]

    print(test)
    s = ir.Schedule(test)
    s.auto_use_lib(ir.CPU())
    print(s.ast())
    print(s.logs())
    assert s.logs() == ["as_matmul(Li)"]


def test_fission_when_prefer_libs():

    @ir.transform
    def test(a, b1, b2, c1, c2):
        ir.declare_var(a, (1000, 1000), "float32", "input", "cpu")
        ir.declare_var(b1, (1000, 1000), "float32", "input", "cpu")
        ir.declare_var(b2, (1000, 1000), "float32", "input", "cpu")
        ir.declare_var(c1, (1000, 1000), "float32", "inout", "cpu")
        ir.declare_var(c2, (1000, 1000), "float32", "inout", "cpu")
        'nid: Li'
        'prefer_libs'
        for i in range(1000):
            for j in range(1000):
                for k in range(1000):
                    c1[i, j] += a[i, k] * b1[k, j]
                    c2[i, j] += a[i, k] * b2[k, j]

    print(test)
    s = ir.Schedule(test)
    s.auto_use_lib(ir.CPU())
    print(s.ast())
    print(s.logs())
    assert "as_matmul(Li.0.lib)" in s.logs()
    assert "as_matmul(Li.1.lib)" in s.logs()
