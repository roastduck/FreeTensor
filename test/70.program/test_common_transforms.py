import freetensor as ft
from freetensor import debug
import numpy as np
import pytest


def test_tiling():
    device = ft.CPU()
    target = device.target()

    with ft.VarDef([
        ("a", (256, 256), "float32", "input", "cpu"),
        ("b", (256, 256), "float32", "input", "cpu"),
        ("c", (256, 256), "float32", "output", "cpu"),
    ]) as (a, b, c):
        with ft.For("i", 0, 256, nid="Li") as i:
            with ft.For("j", 0, 256, nid="Lj") as j:
                with ft.NamedScope("S0"):
                    c[i, j] = 0
                    with ft.For("k", 0, 256, nid="Lk") as k:
                        ft.MarkNid("S1")
                        c[i, j] = c[i, j] + a[i, k] * b[k, j]

    i, j = "Li", "Lj"

    func = ft.Func("main", ["a", "b", "c"], [], ft.pop_ast())
    s = ft.Schedule(func)
    i0, i1 = s.split(i, 32)
    j0, j1 = s.split(j, 32)
    s.reorder([i0, j0, i1, j1])

    s.cache("S0", "c", "cpu")
    s.cache(i1, "a", "cpu")
    s.cache(i1, "b", "cpu")

    func = ft.lower(s.func(), target, verbose=1)

    with ft.VarDef([
        ("a", (256, 256), "float32", "input", "cpu"),
        ("b", (256, 256), "float32", "input", "cpu"),
        ("c", (256, 256), "float32", "output", "cpu"),
    ]) as (a, b, c):
        with ft.For("i.0", 0, 8) as i0:
            with ft.For("j.0", 0, 8) as j0:
                with ft.VarDef("a.r", (32, 256), "float32", "cache",
                               "cpu") as ar:
                    with ft.For("i.1.ar", 32 * i0, 32 * i0 + 32) as i1:
                        with ft.For("k.ar", 0, 256) as k:
                            ft.Any()
                    with ft.VarDef("b.r", (256, 32), "float32", "cache",
                                   "cpu") as br:
                        with ft.For("k.br", 0, 256) as k:
                            with ft.For("j.1.br", 32 * j0, 32 * j0 + 32) as j1:
                                ft.Any()
                        with ft.For("i.1", 0, 32) as i1:
                            with ft.For("j.1", 0, 32) as j1:
                                with ft.VarDef("c.w", (1, 1), "float32",
                                               "cache", "cpu") as cw:
                                    cw[0, 0] = 0
                                    with ft.For("k", 0, 256) as k:
                                        cw[0,
                                           0] = cw[0, 0] + ar[i1, k] * br[k, j1]
                                    c[i1 + 32 * i0, 32 * j0 + j1] = cw[0, 0]
    std = ft.make_reduction(ft.pop_ast())
    assert std.match(func.body)

    code = ft.codegen(func, target, verbose=True)
    a_np = np.random.rand(256, 256).astype("float32")
    b_np = np.random.rand(256, 256).astype("float32")
    c_np = np.zeros((256, 256), dtype="float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np)
    ft.build_binary(code, device)(a=a_arr, b=b_arr, c=c_arr)
    c_np = c_arr.numpy()

    c_std = a_np @ b_np
    assert np.all(np.isclose(c_np, c_std))


def test_tiled_reduction():
    device = ft.CPU()
    target = device.target()

    with ft.VarDef([
        ("x", (256,), "float32", "input", "cpu"),
        ("y", (1,), "float32", "output", "cpu"),
    ]) as (x, y):
        y[0] = 0
        with ft.For("i", 0, 256, nid="Li") as i:
            y[0] = y[0] + x[i]

    i = "Li"

    func = ft.Func("main", ["x", "y"], [], ft.pop_ast())
    s = ft.Schedule(func)
    i0, i1 = s.split(i, 64)
    s.cache_reduction(i1, "y", "cpu")

    func = ft.lower(s.func(), target, verbose=1)

    with ft.VarDef([
        ("x", (256,), "float32", "input", "cpu"),
        ("y", (1,), "float32", "output", "cpu"),
    ]) as (x, y):
        y[0] = 0
        with ft.For("i0", 0, 4) as i0:
            with ft.VarDef("yw", (1,), "float32", "cache", "cpu") as yw:
                yw[0] = 0.0
                with ft.For("i1", 0, 64) as i1:
                    yw[0] = yw[0] + x[i1 + 64 * i0]
                y[0] = y[0] + yw[0]
    std = ft.make_reduction(ft.pop_ast())
    assert std.match(func.body)

    code = ft.codegen(func, target, verbose=True)
    x_np = np.random.rand(256).astype("float32")
    y_np = np.zeros((1,), dtype="float32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.sum(x_np, keepdims=True)
    assert np.all(np.isclose(y_np, y_std))


def test_parallel_reduction():
    device = ft.CPU()
    target = device.target()

    with ft.VarDef([
        ("x", (256,), "float32", "input", "cpu"),
        ("y", (1,), "float32", "output", "cpu"),
    ]) as (x, y):
        ft.MarkNid("S0")
        y[0] = 0
        with ft.For("i", 0, 256, nid="Li") as i:
            y[0] = y[0] + x[i]

    i, S0 = "Li", "S0"

    func = ft.Func("main", ["x", "y"], [], ft.pop_ast())
    s = ft.Schedule(func)
    i0, i1 = s.split(i, 64)
    init, final, _, _ = s.cache_reduction(i1, "y", "cpu")
    final = s.move_to(final, ft.MoveToSide.After, i0)
    S0 = s.move_to(S0, ft.MoveToSide.Before, final)

    s.parallelize(i0, "openmp")

    func = ft.lower(s.func(), target, verbose=1)

    with ft.VarDef([
        ("x", (256,), "float32", "input", "cpu"),
        ("y", (1,), "float32", "output", "cpu"),
    ]) as (x, y):
        with ft.VarDef("yw", (4, 1), "float32", "cache", "cpu") as yw:
            with ft.For("i0", 0, 4) as i0:
                yw[i0, 0] = 0.0
                with ft.For("i1", 0, 64) as i1:
                    yw[i0, 0] = yw[i0, 0] + x[i1 + 64 * i0]
            y[0] = 0
            with ft.For("i0", 0, 4) as i0:
                y[0] = y[0] + yw[i0, 0]
    std = ft.make_reduction(ft.pop_ast())
    assert std.match(func.body)

    code = ft.codegen(func, target, verbose=True)
    x_np = np.random.rand(256).astype("float32")
    y_np = np.zeros((1,), dtype="float32")
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    ft.build_binary(code, device)(x=x_arr, y=y_arr)
    y_np = y_arr.numpy()

    y_std = np.sum(x_np, keepdims=True)
    assert np.all(np.isclose(y_np, y_std))


def test_dynamic_tiling():
    device = ft.CPU()
    target = device.target()

    with ft.VarDef([
        ("n", (), "int32", "input", "byvalue"),
        ("k", (), "int32", "input", "byvalue"),
        ("m", (), "int32", "input", "byvalue"),
    ]) as (n, k, m):
        with ft.VarDef([
            ("a", (n, k), "float32", "input", "cpu"),
            ("b", (k, m), "float32", "input", "cpu"),
            ("c", (n, m), "float32", "output", "cpu"),
        ]) as (a, b, c):
            with ft.For("i", 0, n, nid="Li") as i:
                with ft.For("j", 0, m, nid="Lj") as j:
                    with ft.NamedScope("S0"):
                        c[i, j] = 0
                        with ft.For("p", 0, k, nid="Lp") as p:
                            ft.MarkNid("S1")
                            c[i, j] = c[i, j] + a[i, p] * b[p, j]

    i, j = "Li", "Lj"

    func = ft.Func("main", ["n", "k", "m", "a", "b", "c"], [], ft.pop_ast())
    s = ft.Schedule(func)
    i0, i1 = s.split(i, 32)
    j0, j1 = s.split(j, 32)
    s.reorder([i0, j0, i1, j1])

    s.cache("S0", "c", "cpu")
    s.cache(i1, "a", "cpu")
    s.cache(i1, "b", "cpu")

    s.separate_tail(True)

    func = s.func()
    print(func)
    func = ft.lower(func, target)
    print(func)

    code = ft.codegen(func, target, verbose=True)
    n_np = np.array(300, dtype="int32")
    k_np = np.array(400, dtype="int32")
    m_np = np.array(500, dtype="int32")
    a_np = np.random.rand(300, 400).astype("float32")
    b_np = np.random.rand(400, 500).astype("float32")
    c_np = np.zeros((300, 500), dtype="float32")
    n_arr = ft.Array(n_np)
    k_arr = ft.Array(k_np)
    m_arr = ft.Array(m_np)
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np)
    driver = ft.build_binary(code, device)
    driver(n=n_arr, k=k_arr, m=m_arr, a=a_arr, b=b_arr, c=c_arr)
    c_np = c_arr.numpy()

    c_std = a_np @ b_np
    assert np.all(np.isclose(c_np, c_std))


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_collaborative_fetch():
    device = ft.GPU()
    target = device.target()

    with ft.VarDef([
        ("a", (32, 256), "float32", "input", "gpu/global"),
        ("b", (256, 32), "float32", "input", "gpu/global"),
        ("c", (32, 32), "float32", "output", "gpu/global"),
    ]) as (a, b, c):
        with ft.For("i", 0, 32, nid="Li") as i:
            with ft.For("j", 0, 32, nid="Lj") as j:
                c[i, j] = 0
                with ft.For("k", 0, 256, nid="Lk") as k:
                    c[i, j] = c[i, j] + a[i, k] * b[k, j]

    i, j, k = "Li", "Lj", "Lk"

    func = ft.Func("main", ["a", "b", "c"], [], ft.pop_ast())
    s = ft.Schedule(func)
    k0, k1 = s.split(k, 32)
    fill_a, _, _, _ = s.cache(k1, "a", "gpu/shared")
    fill_b, _, _, _ = s.cache(k1, "b", "gpu/shared")
    s.parallelize(i, "threadIdx.y")
    s.parallelize(j, "threadIdx.x")
    s.parallelize(
        s.find(
            lambda x: x.type() == ft.ASTNodeType.For and x.body.nid == fill_a),
        "threadIdx.x",
    )
    s.parallelize(
        s.find(
            lambda x: x.type() == ft.ASTNodeType.For and x.body.nid == fill_b),
        "threadIdx.y",
    )
    func = ft.lower(s.func(), target, verbose=1)

    code = ft.codegen(func, target, verbose=True)
    a_np = np.random.rand(32, 256).astype("float32")
    b_np = np.random.rand(256, 32).astype("float32")
    c_np = np.zeros((32, 32), dtype="float32")
    a_arr = ft.Array(a_np)
    b_arr = ft.Array(b_np)
    c_arr = ft.Array(c_np)
    ft.build_binary(code, device)(a=a_arr, b=b_arr, c=c_arr)
    c_np = c_arr.numpy()

    c_std = a_np @ b_np
    assert np.all(np.isclose(c_np, c_std))


def test_vectorize_spmv():
    with ft.VarDef([("x1", (64, 64), "int32", "input", "cpu"),
                    ("x2", (64,), "int32", "input", "cpu"),
                    ("y", (64,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ft.For("i", 0, 64, nid="Li") as i:
            ft.MarkNid("S0")
            y[i] = 0
            with ft.For("j", 0, 64, nid="Lj") as j:
                y[i] += x1[i, j] * x2[j]
    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    i0, i1 = s.split("Li", 4)
    s.reorder([i0, "Lj", i1])
    s.move_to("S0", ft.MoveToSide.Before, "Lj")
    s.vectorize(i1)
    s.vectorize(s.find("S0.a").parent_stmt())  # FIXME: do not hard-code S0.a
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast, verbose=1)

    with ft.VarDef([("x1", (64, 64), "int32", "input", "cpu"),
                    ("x2", (64,), "int32", "input", "cpu"),
                    ("y", (64,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ft.For("i0", 0, 16) as i0:
            with ft.For("i1", 0, 4) as i1:
                y[i1 + 4 * i0] = 0
            with ft.For("j", 0, 64, nid="Lj") as j:
                with ft.For("i1", 0, 4) as i1:
                    y[i1 + 4 * i0] += x1[i1 + 4 * i0, j] * x2[j]
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)
