import ir
import numpy as np

target = ir.CPU()
device = ir.Device(target)


def test_multi_level_tiling():
    a = 128
    b = 256
    m = 4
    # c = 64

    @ir.transform
    def test(w, x, y):
        ir.declare_var(w, (m, m, a, b), "int32", "input", "cpu")
        ir.declare_var(x, (m, m, b, a), "int32", "input", "cpu")
        ir.declare_var(y, (m, m, a, a), "int32", "output", "cpu")
        "nid: L1"
        for i in range(m):
            "nid: L2"
            for j in range(m):
                "nid: L3"
                for k in range(b):
                    "nid: L4"
                    for p in range(a):
                        "nid: L5"
                        for q in range(a):
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]

    with ir.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("y.c", (m, m, a, a), "int32", "cache", "cpu")]) as (w, x,
                                                                         y, yc):
        with ir.For("i0", 0, m) as i0:
            with ir.For("i1", 0, m) as i1:
                with ir.For("i2", 0, a) as i2:
                    with ir.For("i3", 0, a) as i3:
                        yc[i0, i1, i2, i3] = y[i0, i1, i2, i3]
        with ir.For("i", 0, m, nid='L1') as i:
            with ir.For("j", 0, m, nid='L2') as j:
                with ir.For("k", 0, b, nid='L3') as k:
                    with ir.For("p", 0, a, nid='L6') as p:
                        with ir.For("q", 0, a, nid='L7') as q:
                            yc[i, j, p,
                               q] = yc[i, j, p,
                                       q] + w[i, j, p, k] * x[i, j, k, q]
        with ir.For("i0", 0, m) as i0:
            with ir.For("i1", 0, m) as i1:
                with ir.For("i2", 0, a) as i2:
                    with ir.For("i3", 0, a) as i3:
                        y[i0, i1, i2, i3] = yc[i0, i1, i2, i3]
    std = ir.pop_ast()
    std = ir.make_reduction(std)

    s = ir.Schedule(test)
    s = ir.AutoSchedule(s, target, device, 8)
    ast = s.test_cache_write()
    assert std.match(ast)
