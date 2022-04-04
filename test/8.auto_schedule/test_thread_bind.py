import ir
import numpy as np

target = ir.GPU()
device = ir.Device(target)


def test_thread_bind():
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
    sch = s.test_thread_bind()
    print(sch.ast())
    std_log = [
        'split(L4, factor=2, nparts=-1)', 'split(L4.0, factor=2, nparts=-1)',
        'split(L4.0.0, factor=2, nparts=-1)',
        'split(L4.0.0.0, factor=2, nparts=-1)',
        'split(L5, factor=2, nparts=-1)', 'split(L5.0, factor=2, nparts=-1)',
        'split(L5.0.0, factor=2, nparts=-1)',
        'split(L5.0.0.0, factor=2, nparts=-1)',
        'split(L3, factor=6, nparts=-1)', 'split(L3.0, factor=6, nparts=-1)',
        'reorder(L4.0.0.0.0, L5.0.0.0.0, L4.0.0.0.1, L5.0.0.0.1, L4.0.0.1, L5.0.0.1, L3.0.0, L3.0.1, L4.0.1, L5.0.1, L3.1, L4.1, L5.1)',
        'split(L6, factor=4, nparts=-1)', 'split(L6.0, factor=2, nparts=-1)',
        'split(L6.0.0, factor=2, nparts=-1)', 'split(L7, factor=4, nparts=-1)',
        'split(L7.0, factor=2, nparts=-1)',
        'split(L7.0.0, factor=2, nparts=-1)',
        'reorder(L6.0.0.0, L7.0.0.0, L6.0.0.1, L7.0.0.1, L6.0.1, L7.0.1, L6.1, L7.1)',
        'fuse(L4.0.0.0.0, L6.0.0.0)', 'fuse(L5.0.0.0.0, L7.0.0.0)',
        'fuse(L4.0.0.0.1, L6.0.0.1)', 'fuse(L5.0.0.0.1, L7.0.0.1)',
        'fuse(L4.0.0.1, L6.0.1)', 'fuse(L5.0.0.1, L7.0.1)', 'cache(#34, y)',
        'merge(fused.L4.0.0.0.0.L6.0.0.0, fused.L5.0.0.0.0.L7.0.0.0)',
        'merge(fused.L4.0.0.0.1.L6.0.0.1, fused.L5.0.0.0.1.L7.0.0.1)',
        'merge(fused.L4.0.0.1.L6.0.1, fused.L5.0.0.1.L7.0.1)',
        'parallelize(merged.fused.L4.0.0.0.0.L6.0.0.0.fused.L5.0.0.0.0.L7.0.0.0, blockIdx.x)',
        'blend(merged.fused.L4.0.0.0.1.L6.0.0.1.fused.L5.0.0.0.1.L7.0.0.1)',
        'parallelize(merged.fused.L4.0.0.1.L6.0.1.fused.L5.0.0.1.L7.0.1, threadIdx.x)'
    ]
    sch_log = sch.logs()
    assert len(sch_log) == len(std_log)
    for l, r in zip(sch_log, std_log):
        if l.startswith('cache'):
            assert r.startswith('cache')
        else:
            assert l == r
