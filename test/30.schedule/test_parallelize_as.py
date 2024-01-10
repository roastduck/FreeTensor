import freetensor as ft
import pytest


def test_partitioned_by_tile():
    with ft.VarDef([("a", (8,), "int32", "input", "cpu"),
                    ("c", (8,), "int32", "output", "cpu")]) as (a, c):
        ft.MarkLabel("Vb")
        with ft.VarDef("b", (8,), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", 0, 2) as j:
                    b[i * 2 + j] = a[i * 2 + j] * 2
            with ft.For("i", 0, 8, label="L2") as i:
                c[i] = b[i] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=1)
    s.parallelize("L1", "openmp")
    s.parallelize_as("L2", "L1", "Vb")
    ast = s.ast()
    assert ft.find_stmt(ast, "<For>->L2").property.parallel == "openmp"

    with ft.VarDef([("a", (8,), "int32", "input", "cpu"),
                    ("c", (8,), "int32", "output", "cpu")]) as (a, c):
        ft.MarkLabel("Vb")
        with ft.VarDef("b", (8,), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", 0, 2) as j:
                    b[i * 2 + j] = a[i * 2 + j] * 2
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", i * 2, i * 2 + 2) as j:
                    c[j] = b[j] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_partitioned_by_stride():
    with ft.VarDef([("a", (8,), "int32", "input", "cpu"),
                    ("c", (8,), "int32", "output", "cpu")]) as (a, c):
        ft.MarkLabel("Vb")
        with ft.VarDef("b", (8,), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", 0, 2) as j:
                    b[j * 4 + i] = a[j * 4 + i] * 2
            with ft.For("i", 0, 8, label="L2") as i:
                c[i] = b[i] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=1)
    s.parallelize("L1", "openmp")
    s.parallelize_as("L2", "L1", "Vb")
    ast = s.ast()
    assert ft.find_stmt(ast, "<For>->L2").property.parallel == "openmp"

    with ft.VarDef([("a", (8,), "int32", "input", "cpu"),
                    ("c", (8,), "int32", "output", "cpu")]) as (a, c):
        ft.MarkLabel("Vb")
        with ft.VarDef("b", (8,), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", 0, 2) as j:
                    b[j * 4 + i] = a[j * 4 + i] * 2
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", i, i + 5, 4) as j:
                    c[j] = b[j] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_reference_after_nest():
    with ft.VarDef([("a", (8,), "int32", "input", "cpu"),
                    ("c", (8,), "int32", "output", "cpu")]) as (a, c):
        ft.MarkLabel("Vb")
        with ft.VarDef("b", (8,), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 8, label="L2") as i:
                b[i] = a[i] + 1
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", 0, 2) as j:
                    c[i * 2 + j] = b[i * 2 + j] * 2
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=1)
    s.parallelize("L1", "openmp")
    s.parallelize_as("L2", "L1", "Vb")
    ast = s.ast()
    assert ft.find_stmt(ast, "<For>->L2").property.parallel == "openmp"

    with ft.VarDef([("a", (8,), "int32", "input", "cpu"),
                    ("c", (8,), "int32", "output", "cpu")]) as (a, c):
        ft.MarkLabel("Vb")
        with ft.VarDef("b", (8,), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", i * 2, i * 2 + 2) as j:
                    b[j] = a[j] + 1
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", 0, 2) as j:
                    c[i * 2 + j] = b[i * 2 + j] * 2
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_multiple_levels():
    with ft.VarDef([("a", (128, 128), "int32", "input", "cpu"),
                    ("c", (128, 128), "int32", "output", "cpu")]) as (a, c):
        ft.MarkLabel("Vb")
        with ft.VarDef("b", (128, 128), "int32", "cache", "cpu") as b:
            with ft.For("i0", 0, 8, label="L1i") as i0:
                with ft.For("j0", 0, 8, label="L1j") as j0:
                    with ft.For("i", 16 * i0, 16 * i0 + 16) as i:
                        with ft.For("j", 16 * j0, 16 * j0 + 16) as j:
                            b[i, j] = a[i, j] * 2
            with ft.For("i", 0, 128, label="L2") as i:
                with ft.For("j", 0, 128) as j:
                    c[i, j] = b[i, j] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=1)
    s.parallelize("L1i", "blockIdx.x")
    s.parallelize("L1j", "threadIdx.x")
    s.parallelize_as("L2", "L1i", "Vb")
    ast = s.ast()
    assert ft.find_stmt(ast, "<For>->L2").property.parallel == "threadIdx.x"
    assert ft.find_stmt(ast,
                        "<For>-><For>->L2").property.parallel == "blockIdx.x"

    with ft.VarDef([("a", (128, 128), "int32", "input", "cpu"),
                    ("c", (128, 128), "int32", "output", "cpu")]) as (a, c):
        ft.MarkLabel("Vb")
        with ft.VarDef("b", (128, 128), "int32", "cache", "cpu") as b:
            with ft.For("i0", 0, 8, label="L1i") as i0:
                with ft.For("j0", 0, 8, label="L1j") as j0:
                    with ft.For("i", 16 * i0, 16 * i0 + 16) as i:
                        with ft.For("j", 16 * j0, 16 * j0 + 16) as j:
                            b[i, j] = a[i, j] * 2
            with ft.For("i0", 0, 8, label="L1i") as i0:
                with ft.For("j0", 0, 8, label="L1j") as j0:
                    with ft.For("i", 16 * i0, 16 * i0 + 16) as i:
                        with ft.For("j", 16 * j0, 16 * j0 + 16) as j:
                            c[i, j] = b[i, j] + 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_reject_thread_non_local_reference_shared_by_loop():
    with ft.VarDef([("a", (8,), "int32", "input", "cpu"),
                    ("c", (2,), "int32", "output", "cpu")]) as (a, c):
        ft.MarkLabel("Vb")
        with ft.VarDef("b", (2,), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", 0, 2) as j:
                    b[j] += a[i * 2 + j] * 2
            with ft.For("i", 0, 2, label="L2") as i:
                c[i] = b[i] + 1
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=1)
    s.parallelize("L1", "openmp")
    with pytest.raises(ft.InvalidSchedule):
        s.parallelize_as("L2", "L1", "Vb")


def test_reject_thread_non_local_reference_shared_by_multiple_access_sites():
    with ft.VarDef([("a", (8,), "int32", "input", "cpu"),
                    ("c", (8,), "int32", "output", "cpu")]) as (a, c):
        ft.MarkLabel("Vb")
        with ft.VarDef("b", (8,), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 8, label="L2") as i:
                b[i] = a[i] + 1
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", 0, 2) as j:
                    c[i * 2 + j] = b[i * 2 + j] + b[(i * 2 + j + 1) % 8]
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=1)
    s.parallelize("L1", "openmp")
    with pytest.raises(ft.InvalidSchedule):
        s.parallelize_as("L2", "L1", "Vb")


def test_reject_thread_non_local_destination_nest():
    with ft.VarDef([("a", (8,), "int32", "input", "cpu"),
                    ("c", (8,), "int32", "output", "cpu")]) as (a, c):
        ft.MarkLabel("Vb")
        with ft.VarDef("b", (8,), "int32", "cache", "cpu") as b:
            with ft.For("i", 0, 4, label="L1") as i:
                with ft.For("j", 0, 2) as j:
                    b[i * 2 + j] = a[i * 2 + j] * 2
            with ft.For("i", 0, 8, label="L2") as i:
                c[i] = b[i] + b[(i + 1) % 8]
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast, verbose=1)
    s.parallelize("L1", "openmp")
    with pytest.raises(ft.InvalidSchedule):
        s.parallelize_as("L2", "L1", "Vb")
