import ir
import ir.debug
import numpy as np

def test_tiling():
    target = ir.CPU()
    device = ir.Device(target)

    with ir.VarDef([
            ("a", (256, 256), "float32", "input", "cpu"),
            ("b", (256, 256), "float32", "input", "cpu"),
            ("c", (256, 256), "float32", "output", "cpu")]) as (a, b, c):
        with ir.For("i", 0, 256, nid="Li") as i:
            with ir.For("j", 0, 256, nid="Lj") as j:
                with ir.NamedScope("S0"):
                    c[i, j] = 0
                    with ir.For("k", 0, 256, nid="Lk") as k:
                        ir.MarkNid("S1")
                        c[i, j] = c[i, j] + a[i, k] * b[k, j]

    i, j = "Li", "Lj"

    s = ir.Schedule(ir.pop_ast())
    i0, i1 = s.split(i, 32)
    j0, j1 = s.split(j, 32)
    s.reorder([i0, j0, i1, j1])

    s.cache("S0", "c", "cpu")
    s.cache(i1, "a", "cpu")
    s.cache(i1, "b", "cpu")

    ast = ir.lower(s.ast(), target)
    print(ast)

    with ir.VarDef([
            ("a", (256, 256), "float32", "input", "cpu"),
            ("b", (256, 256), "float32", "input", "cpu"),
            ("c", (256, 256), "float32", "output", "cpu")]) as (a, b, c):
        with ir.For("i.0", 0, 8) as i0:
            with ir.For("j.0", 0, 8) as j0:
                with ir.VarDef("a.r", (32, 256), "float32", ir.AccessType.Cache, "cpu") as ar:
                    with ir.For("i.1.ar", 32 * i0, 32 * i0 + 32) as i1:
                        with ir.For("k.ar", 0, 256) as k:
                            ir.Any()
                    with ir.VarDef("b.r", (256, 32), "float32", ir.AccessType.Cache, "cpu") as br:
                        with ir.For("k.br", 0, 256) as k:
                            with ir.For("j.1.br", 32 * j0, 32 * j0 + 32) as j1:
                                ir.Any()
                        with ir.For("i.1", 0, 32) as i1:
                            with ir.For("j.1", 0, 32) as j1:
                                with ir.VarDef("c.w", (1, 1), "float32", ir.AccessType.Cache, "cpu") as cw:
                                    cw[0, 0] = 0
                                    with ir.For("k", 0, 256) as k:
                                        cw[0, 0] = cw[0, 0] + ar[i1, k] * br[k, j1]
                                    c[i1 + 32 * i0, 32 * j0 + j1] = cw[0, 0]
    std = ir.make_reduction(ir.pop_ast())
    assert std.match(ast)

    code, params = ir.codegen(ast, target)
    print(code)
    a_np = np.random.rand(256, 256).astype("float32")
    b_np = np.random.rand(256, 256).astype("float32")
    c_np = np.zeros((256, 256), dtype="float32")
    a_arr = ir.Array(a_np, device)
    b_arr = ir.Array(b_np, device)
    c_arr = ir.Array(c_np, device)
    driver = ir.Driver(code, params, device)
    driver.set_params({"a": a_arr, "b": b_arr, "c": c_arr})
    driver.run()
    c_np = c_arr.numpy().reshape(256, 256)

    c_std = a_np @ b_np
    assert np.all(np.isclose(c_np, c_std))

def test_tiled_reduction():
    target = ir.CPU()
    device = ir.Device(target)

    with ir.VarDef([
            ("x", (256,), "float32", "input", "cpu"),
            ("y", (1,), "float32", "output", "cpu")]) as (x, y):
        y[0] = 0
        with ir.For("i", 0, 256, nid="Li") as i:
            y[0] = y[0] + x[i]

    i = "Li"

    s = ir.Schedule(ir.pop_ast())
    i0, i1 = s.split(i, 64)
    s.cache_reduction(i1, "y", "cpu")

    ast = ir.lower(s.ast(), target)
    print(ast)

    with ir.VarDef([
            ("x", (256,), "float32", "input", "cpu"),
            ("y", (1,), "float32", "output", "cpu")]) as (x, y):
        y[0] = 0
        with ir.For("i0", 0, 4) as i0:
            with ir.VarDef("yw", (1,), "float32", "cache", "cpu") as yw:
                yw[0] = 0.
                with ir.For("i1", 0, 64) as i1:
                    yw[0] = yw[0] + x[i1 + 64 * i0]
                y[0] = y[0] + yw[0]
    std = ir.make_reduction(ir.pop_ast())
    assert std.match(ast)

    code, params = ir.codegen(ast, target)
    print(code)
    x_np = np.random.rand(256).astype("float32")
    y_np = np.zeros((1,), dtype="float32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(code, params, device)
    driver.set_params({"x": x_arr, "y": y_arr})
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.sum(x_np, keepdims=True)
    assert np.all(np.isclose(y_np, y_std))

def test_parallel_reduction():
    target = ir.CPU()
    device = ir.Device(target)

    with ir.VarDef([
            ("x", (256,), "float32", "input", "cpu"),
            ("y", (1,), "float32", "output", "cpu")]) as (x, y):
        ir.MarkNid("S0")
        y[0] = 0
        with ir.For("i", 0, 256, nid="Li") as i:
            y[0] = y[0] + x[i]

    i, S0 = "Li", "S0"

    s = ir.Schedule(ir.pop_ast())
    i0, i1 = s.split(i, 64)
    init, final, _ = s.cache_reduction(i1, "y", "cpu")
    final = s.move_to(final, ir.MoveToSide.After, i0)
    S0 = s.move_to(S0, ir.MoveToSide.Before, final)

    s.parallelize(i0, "openmp")

    ast = ir.lower(s.ast(), target)
    print(ast)

    with ir.VarDef([
            ("x", (256,), "float32", "input", "cpu"),
            ("y", (1,), "float32", "output", "cpu")]) as (x, y):
        with ir.VarDef("yw", (4, 1), "float32", "cache", "cpu") as yw:
            with ir.For("i0", 0, 4) as i0:
                yw[i0, 0] = 0.
                with ir.For("i1", 0, 64) as i1:
                    yw[i0, 0] = yw[i0, 0] + x[i1 + 64 * i0]
            y[0] = 0
            with ir.For("i0", 0, 4) as i0:
                y[0] = y[0] + yw[i0, 0]
    std = ir.make_reduction(ir.pop_ast())
    assert std.match(ast)

    code, params = ir.codegen(ast, target)
    print(code)
    x_np = np.random.rand(256).astype("float32")
    y_np = np.zeros((1,), dtype="float32")
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    driver = ir.Driver(code, params, device)
    driver.set_params({"x": x_arr, "y": y_arr})
    driver.run()
    y_np = y_arr.numpy()

    y_std = np.sum(x_np, keepdims=True)
    assert np.all(np.isclose(y_np, y_std))

def test_dynamic_tiling():
    target = ir.CPU()
    device = ir.Device(target)
    host = device

    with ir.VarDef([
            ("n", (), "int32", "input", "byvalue"),
            ("k", (), "int32", "input", "byvalue"),
            ("m", (), "int32", "input", "byvalue")]) as (n, k, m):
        with ir.VarDef([
                ("a", (n[()], k[()]), "float32", "input", "cpu"),
                ("b", (k[()], m[()]), "float32", "input", "cpu"),
                ("c", (n[()], m[()]), "float32", "output", "cpu")]) as (a, b, c):
            with ir.For("i", 0, n[()], nid="Li") as i:
                with ir.For("j", 0, m[()], nid="Lj") as j:
                    with ir.NamedScope("S0"):
                        c[i, j] = 0
                        with ir.For("p", 0, k[()], nid="Lp") as p:
                            ir.MarkNid("S1")
                            c[i, j] = c[i, j] + a[i, p] * b[p, j]

    i, j = "Li", "Lj"

    s = ir.Schedule(ir.pop_ast())
    i0, i1 = s.split(i, 32)
    j0, j1 = s.split(j, 32)
    s.reorder([i0, j0, i1, j1])

    s.cache("S0", "c", "cpu")
    s.cache(i1, "a", "cpu")
    s.cache(i1, "b", "cpu")

    ast = s.ast()
    print(ast)
    ast = ir.lower(ast, target)
    print(ast)

    code, params = ir.codegen(ast, target)
    print(code)
    n_np = np.array(300, dtype="int32")
    k_np = np.array(400, dtype="int32")
    m_np = np.array(500, dtype="int32")
    a_np = np.random.rand(300, 400).astype("float32")
    b_np = np.random.rand(400, 500).astype("float32")
    c_np = np.zeros((300, 500), dtype="float32")
    n_arr = ir.Array(n_np, host)
    k_arr = ir.Array(k_np, host)
    m_arr = ir.Array(m_np, host)
    a_arr = ir.Array(a_np, device)
    b_arr = ir.Array(b_np, device)
    c_arr = ir.Array(c_np, device)
    driver = ir.Driver(code, params, device)
    driver.set_params({
        "n": n_arr, "k": k_arr, "m": m_arr,
        "a": a_arr, "b": b_arr, "c": c_arr})
    driver.run()
    c_np = c_arr.numpy().reshape(300, 500)

    c_std = a_np @ b_np
    assert np.all(np.isclose(c_np, c_std))

def test_collaborative_fetch():
    target = ir.GPU()
    device = ir.Device(target)
    host = ir.Device(ir.CPU())

    with ir.VarDef([
            ("a", (32, 256), "float32", "input", "cpu"),
            ("b", (256, 32), "float32", "input", "cpu"),
            ("c", (32, 32), "float32", "output", "cpu")]) as (a, b, c):
        with ir.For("i", 0, 32, nid="Li") as i:
            with ir.For("j", 0, 32, nid="Lj") as j:
                c[i, j] = 0
                with ir.For("k", 0, 256, nid="Lk") as k:
                    c[i, j] = c[i, j] + a[i, k] * b[k, j]

    i, j, k = "Li", "Lj", "Lk"

    s = ir.Schedule(ir.pop_ast())
    k0, k1 = s.split(k, 32)
    fill_a, _, _ = s.cache(k1, "a", "gpu/shared")
    fill_b, _, _ = s.cache(k1, "b", "gpu/shared")
    s.parallelize(i, "threadIdx.y")
    s.parallelize(j, "threadIdx.x")
    s.parallelize(s.find(lambda x: x.type() == "For" and x.node().body.nid == fill_a), "threadIdx.x")
    s.parallelize(s.find(lambda x: x.type() == "For" and x.node().body.nid == fill_b), "threadIdx.y")
    ast = ir.lower(s.ast(), target)
    print(ast)

    code, params = ir.codegen(ast, target)
    print(ir.debug.with_line_no(code))
    a_np = np.random.rand(32, 256).astype("float32")
    b_np = np.random.rand(256, 32).astype("float32")
    c_np = np.zeros((32, 32), dtype="float32")
    a_arr = ir.Array(a_np, device)
    b_arr = ir.Array(b_np, device)
    c_arr = ir.Array(c_np, device)
    driver = ir.Driver(code, params, device)
    driver.set_params({"a": a_arr, "b": b_arr, "c": c_arr})
    driver.run()
    c_np = c_arr.numpy().reshape(32, 32)

    c_std = a_np @ b_np
    assert np.all(np.isclose(c_np, c_std))

