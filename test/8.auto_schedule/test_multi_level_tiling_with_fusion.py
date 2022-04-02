import ir
import numpy as np

target = ir.CPU()
device = ir.Device(target)


def test_fusion():
    a = 128
    b = 256
    m = 4

    @ir.transform
    def test(w, x, y, z, u):
        ir.declare_var(w, (m, m, a, b), "int32", "input", "cpu")
        ir.declare_var(x, (m, m, b, a), "int32", "input", "cpu")
        ir.declare_var(y, (1, 1, a, a), "int32", "cache", "cpu")
        ir.declare_var(z, (m, m, a, a), "int32", "output", "cpu")
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
                            y[1, 1, p,
                              q] = y[1, 1, p, q] + w[i, j, p, k] * x[i, j, k, q]
                "nid: L6"
                for p in range(a):
                    "nid: L7"
                    for q in range(a):
                        z[i, j, p, q] = y[1, 1, p, q]

    s = ir.Schedule(test)
    print(s.ast())
    s = ir.AutoSchedule(s, target, device, 8)
    ast = s.test_multi_level_tiling_with_fusion(1)
    print(ast)


def test_cache():
    a = 128
    b = 256
    m = 4

    @ir.transform
    def test(y, z):
        # ir.declare_var(w, (a, b), "int32", "input", "cpu")
        # ir.declare_var(x, (b, a), "int32", "input", "cpu")
        ir.declare_var(y, (a,), "int32", "cache", "cpu")
        ir.declare_var(z, (a,), "int32", "output", "cpu")
        "nid: L1"
        for p1 in range(4):
            "nid: L4"
            for p in range(a // 4):
                y[p + p1 * 32] = p + p1 * 32
            "nid: L6"
            for p0 in range(a // 4):
                z[p0 + p1 * 32] = y[p0 + p1 * 32]

    s = ir.Schedule(test)
    print(s.ast())
    s.cache(s.find("L1").node().body, "y", "cpu")
    print(s.ast())
