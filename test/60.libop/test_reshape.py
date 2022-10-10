import numpy as np

import freetensor as ft
from freetensor import libop


def test_basic():

    @ft.lower(skip_passes=["use_builtin_div"], verbose=1)
    @ft.transform(verbose=1)
    def f(x, y):
        x: ft.Var[(3, 5), "float32", "input", "cpu"]
        y: ft.Var[(5, 3), "float32", "output", "cpu"]
        #! label: reshape
        libop.reshape_(x, y)

    with ft.VarDef([("x", (3, 5), "float32", "input", "cpu"),
                    ("y", (5, 3), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 15) as i:
            y[i // 3, i % 3] = x[i // 5, i % 5]
    std = ft.pop_ast()
    assert std.match(f.body)

    f = ft.build_binary(ft.codegen(f))

    x_np = np.random.rand(3, 5).astype("float32")
    x_arr = ft.Array(x_np)
    y_np = np.zeros((5, 3), dtype="float32")
    y_arr = ft.Array(y_np)
    f(x_arr, y_arr)
    y_np = y_arr.numpy()

    assert np.all(y_np == x_np.reshape(5, 3))


def test_split_dim_with_multiple_loops():

    @ft.lower(skip_passes=["use_builtin_div"], verbose=1)
    @ft.transform(verbose=1)
    def f(x, y):
        x: ft.Var[(15,), "float32", "input", "cpu"]
        y: ft.Var[(5, 3), "float32", "output", "cpu"]
        #! label: reshape
        libop.reshape_(x, y)

    with ft.VarDef([("x", (15,), "float32", "input", "cpu"),
                    ("y", (5, 3), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 5) as i:
            with ft.For("j", 0, 3) as j:
                y[i, j] = x[i * 3 + j]
    std = ft.pop_ast()
    assert std.match(f.body)

    f = ft.build_binary(ft.codegen(f))

    x_np = np.random.rand(15).astype("float32")
    x_arr = ft.Array(x_np)
    y_np = np.zeros((5, 3), dtype="float32")
    y_arr = ft.Array(y_np)
    f(x_arr, y_arr)
    y_np = y_arr.numpy()

    assert np.all(y_np == x_np.reshape(5, 3))


def test_merge_dims_with_multiple_loops():

    @ft.lower(skip_passes=["use_builtin_div"], verbose=1)
    @ft.transform(verbose=1)
    def f(x, y):
        x: ft.Var[(3, 5), "float32", "input", "cpu"]
        y: ft.Var[(15,), "float32", "output", "cpu"]
        #! label: reshape
        libop.reshape_(x, y)

    with ft.VarDef([("x", (3, 5), "float32", "input", "cpu"),
                    ("y", (15,), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 3) as i:
            with ft.For("j", 0, 5) as j:
                y[i * 5 + j] = x[i, j]
    std = ft.pop_ast()
    assert std.match(f.body)

    f = ft.build_binary(ft.codegen(f))

    x_np = np.random.rand(3, 5).astype("float32")
    x_arr = ft.Array(x_np)
    y_np = np.zeros((15,), dtype="float32")
    y_arr = ft.Array(y_np)
    f(x_arr, y_arr)
    y_np = y_arr.numpy()

    assert np.all(y_np == x_np.reshape(15,))


def test_non_affecting_dims_in_different_loops():

    @ft.lower(skip_passes=["use_builtin_div"], verbose=1)
    @ft.transform(verbose=1)
    def f(x, y):
        x: ft.Var[(3, 5, 6), "float32", "input", "cpu"]
        y: ft.Var[(5, 3, 6), "float32", "output", "cpu"]
        #! label: reshape
        libop.reshape_(x, y)

    with ft.VarDef([("x", (3, 5, 6), "float32", "input", "cpu"),
                    ("y", (5, 3, 6), "float32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 15) as i:
            with ft.For("j", 0, 6) as j:
                y[i // 3, i % 3, j] = x[i // 5, i % 5, j]
    std = ft.pop_ast()
    assert std.match(f.body)

    f = ft.build_binary(ft.codegen(f))

    x_np = np.random.rand(3, 5, 6).astype("float32")
    x_arr = ft.Array(x_np)
    y_np = np.zeros((5, 3, 6), dtype="float32")
    y_arr = ft.Array(y_np)
    f(x_arr, y_arr)
    y_np = y_arr.numpy()

    assert np.all(y_np == x_np.reshape(5, 3, 6))


def test_split_symbolic():

    @ft.lower(skip_passes=["use_builtin_div"], verbose=1)
    @ft.transform(verbose=1)
    def f(n, m, x, y):
        n: ft.Var[(), "int32", "input", "cpu"]
        m: ft.Var[(), "int32", "input", "cpu"]
        x: ft.Var[(n[...] * m[...],), "float32", "input", "cpu"]
        y: ft.Var[(n[...], m[...]), "float32", "output", "cpu"]
        #! label: reshape
        libop.reshape_(x, y)

    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("m", (), "int32", "input", "cpu")]) as (n, m):
        with ft.VarDef([("x", (n[...] * m[...],), "float32", "input", "cpu"),
                        ("y", (n[...], m[...]), "float32", "output", "cpu")
                       ]) as (x, y):
            with ft.For("i", 0, n[...]) as i:
                with ft.For("j", 0, m[...]) as j:
                    y[i, j] = x[i * m[...] + j]
    std = ft.pop_ast()
    assert std.match(f.body)

    f = ft.build_binary(ft.codegen(f))

    x_np = np.random.rand(15).astype("float32")
    x_arr = ft.Array(x_np)
    y_np = np.zeros((5, 3), dtype="float32")
    y_arr = ft.Array(y_np)
    f(np.array(5, dtype="int32"), np.array(3, dtype="int32"), x_arr, y_arr)
    y_np = y_arr.numpy()

    assert np.all(y_np == x_np.reshape(5, 3))


def test_merge_symbolic():

    @ft.lower(skip_passes=["use_builtin_div"], verbose=1)
    @ft.transform(verbose=1)
    def f(n, m, x, y):
        n: ft.Var[(), "int32", "input", "cpu"]
        m: ft.Var[(), "int32", "input", "cpu"]
        x: ft.Var[(n[...], m[...]), "float32", "input", "cpu"]
        y: ft.Var[(n[...] * m[...],), "float32", "output", "cpu"]
        #! label: reshape
        libop.reshape_(x, y)

    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("m", (), "int32", "input", "cpu")]) as (n, m):
        with ft.VarDef([("x", (n[...], m[...]), "float32", "input", "cpu"),
                        ("y", (n[...] * m[...],), "float32", "output", "cpu")
                       ]) as (x, y):
            with ft.For("i", 0, n[...]) as i:
                with ft.For("j", 0, m[...]) as j:
                    y[i * m[...] + j] = x[i, j]
    std = ft.pop_ast()
    assert std.match(f.body)

    f = ft.build_binary(ft.codegen(f))

    x_np = np.random.rand(5, 3).astype("float32")
    x_arr = ft.Array(x_np)
    y_np = np.zeros((15,), dtype="float32")
    y_arr = ft.Array(y_np)
    f(np.array(5, dtype="int32"), np.array(3, dtype="int32"), x_arr, y_arr)
    y_np = y_arr.numpy()

    assert np.all(y_np == x_np.reshape(15))


def test_non_affecting_dims_in_different_loops_symbolic():

    @ft.lower(skip_passes=["use_builtin_div"], verbose=1)
    @ft.transform(verbose=1)
    def f(n, m, k, x, y):
        n: ft.Var[(), "int32", "input", "cpu"]
        m: ft.Var[(), "int32", "input", "cpu"]
        k: ft.Var[(), "int32", "input", "cpu"]
        x: ft.Var[(n[...], m[...], k[...]), "float32", "input", "cpu"]
        y: ft.Var[(m[...], n[...], k[...]), "float32", "output", "cpu"]
        #! label: reshape
        libop.reshape_(x, y)

    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("m", (), "int32", "input", "cpu"),
                    ("k", (), "int32", "input", "cpu")]) as (n, m, k):
        with ft.VarDef([
            ("x", (n[...], m[...], k[...]), "float32", "input", "cpu"),
            ("y", (m[...], n[...], k[...]), "float32", "output", "cpu")
        ]) as (x, y):
            with ft.For("i", 0, n[...] * m[...]) as i:
                with ft.For("j", 0, k[...]) as j:
                    # TODO: (i // n[...]) % m[...] should be i // n[...]
                    y[(i // n[...]) % m[...], i % n[...],
                      j] = x[(i // m[...]) % n[...], i % m[...], j]
    std = ft.pop_ast()
    assert std.match(f.body)

    f = ft.build_binary(ft.codegen(f))

    x_np = np.random.rand(5, 3, 10).astype("float32")
    x_arr = ft.Array(x_np)
    y_np = np.zeros((3, 5, 10), dtype="float32")
    y_arr = ft.Array(y_np)
    f(np.array(5, dtype="int32"), np.array(3, dtype="int32"),
      np.array(10, dtype="int32"), x_arr, y_arr)
    y_np = y_arr.numpy()

    assert np.all(y_np == x_np.reshape(3, 5, 10))


def test_out_of_place():

    @ft.lower(skip_passes=["use_builtin_div"], verbose=1)
    @ft.transform(verbose=1)
    def f(x: ft.Var[(3, 5), "float32", "input", "cpu"]):
        #! label: reshape
        return libop.reshape(x, [5, 3])

    f = ft.build_binary(ft.codegen(f))

    x_np = np.random.rand(3, 5).astype("float32")
    y_np = f(x_np).numpy()

    assert np.all(y_np == x_np.reshape(5, 3))
