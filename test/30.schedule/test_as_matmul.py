import freetensor as ft
import pytest

# This file only tests for invalid cases. Valid cases are tested per backend
# in test/40.codegen


def test_not_plain_iterator():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 64), "float32", "input", "cpu"]
        b: ft.Var[(64, 72), "float32", "input", "cpu"]
        c: ft.Var[(48, 72), "float32", "inout", "cpu"]
        #! label: L1
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[i * i, k] * b[k, j]

    s = ft.Schedule(test)
    with pytest.raises(ft.InvalidSchedule):
        s.as_matmul("L1")


def test_splitted_dim_not_contiguous_iterating_range():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 16, 4), "float32", "input", "cpu"]
        b: ft.Var[(16, 4, 72), "float32", "input", "cpu"]
        c: ft.Var[(48, 72), "float32", "inout", "cpu"]
        #! label: L1
        for k0 in range(16):
            for i in range(48):
                for j in range(72):
                    for k1 in range(3):  # Invalid because not 4
                        c[i, j] += a[i, k0, k1] * b[k0, k1, j]

    s = ft.Schedule(test)
    with pytest.raises(ft.InvalidSchedule):
        s.as_matmul("L1")


def test_splitted_dim_not_contiguous_indices():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(16, 48, 4), "float32", "input", "cpu"]
        b: ft.Var[(16, 4, 72), "float32", "input", "cpu"]
        c: ft.Var[(48, 72), "float32", "inout", "cpu"]
        #! label: L1
        for k0 in range(16):
            for i in range(48):
                for j in range(72):
                    for k1 in range(4):
                        # Invalid because not a[i, k0, k1]
                        c[i, j] += a[k0, i, k1] * b[k0, k1, j]

    s = ft.Schedule(test)
    with pytest.raises(ft.InvalidSchedule):
        s.as_matmul("L1")


def test_splitted_dim_different_order():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 16, 4), "float32", "input", "cpu"]
        b: ft.Var[(4, 16, 72), "float32", "input", "cpu"]
        c: ft.Var[(48, 72), "float32", "inout", "cpu"]
        #! label: L1
        for k0 in range(16):
            for i in range(48):
                for j in range(72):
                    for k1 in range(4):
                        # Invalid because (k0, k1) != (k1, k0)
                        c[i, j] += a[i, k0, k1] * b[k1, k0, j]

    s = ft.Schedule(test)
    with pytest.raises(ft.InvalidSchedule):
        s.as_matmul("L1")


def test_not_reduce_add():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 64), "float32", "input", "cpu"]
        b: ft.Var[(64, 72), "float32", "input", "cpu"]
        c: ft.Var[(48, 72), "float32", "inout", "cpu"]
        #! label: L1
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[i, j] *= a[i, k] * b[k, j]

    s = ft.Schedule(test)
    with pytest.raises(ft.InvalidSchedule):
        s.as_matmul("L1")


def test_not_mul():

    @ft.transform
    def test(a, b, c):
        a: ft.Var[(48, 64), "float32", "input", "cpu"]
        b: ft.Var[(64, 72), "float32", "input", "cpu"]
        c: ft.Var[(48, 72), "float32", "inout", "cpu"]
        #! label: L1
        for i in range(48):
            for j in range(72):
                for k in range(64):
                    c[i, j] += a[i, k] + b[k, j]

    s = ft.Schedule(test)
    with pytest.raises(ft.InvalidSchedule):
        s.as_matmul("L1")
