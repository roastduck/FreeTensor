import freetensor as ft
import numpy as np

target = ft.CPU()
device = ft.Device(target.type())


def test_cache_write():
    a = 128
    b = 256
    m = 4
    # c = 64

    @ft.transform
    def test(w, x, y, z):
        w: ft.Var[(m, m, a, b), "int32", "input", "cpu"]
        x: ft.Var[(m, m, b, a), "int32", "input", "cpu"]
        y: ft.Var[(1, 1, a, a), "int32", "inout", "cpu"]
        z: ft.Var[(m, m, a, a), "int32", "output", "cpu"]
        #! nid: L1
        for i in range(m):
            #! nid: L2
            for j in range(m):
                #! nid: L4
                for p in range(a):
                    #! nid: L5
                    for q in range(a):
                        #! nid: Init
                        y[0, 0, p, q] = 0
                        #! nid: L3
                        for k in range(b):
                            y[0, 0, p,
                              q] = y[0, 0, p, q] + w[i, j, p, k] * x[i, j, k, q]
                #! nid: L6
                for p in range(a):
                    #! nid: L7
                    for q in range(a):
                        z[i, j, p, q] = y[0, 0, p, q]

    s = ft.Schedule(test)
    s = ft.AutoSchedule(s,
                        target,
                        device,
                        rule_set={"multi_level_tiling", "parallelize"})
    sch = s.test_round()
    func = ft.lower(sch.func(), target)
    print(func)
    code = ft.codegen(func, target, verbose=True)
    w_np = np.zeros((m, m, a, b), dtype="int32")
    x_np = np.zeros((m, m, b, a), dtype="int32")
    y_np = np.zeros((1, 1, a, a), dtype="int32")
    z_np = np.zeros((m, m, a, a), dtype="int32")
    w_arr = ft.Array(w_np)
    x_arr = ft.Array(x_np)
    y_arr = ft.Array(y_np)
    z_arr = ft.Array(z_np)
    ft.build_binary(code, device)(w=w_arr, x=x_arr, y=y_arr, z=z_arr)
    std_log = [
        'split(L4, 4, -1, 0)', 'split(L4.0, 2, -1, 0)', 'split(L5, 4, -1, 0)',
        'split(L5.0, 2, -1, 0)',
        'reorder(L4.0.0, L5.0.0, L4.0.1, L5.0.1, L4.1, L5.1)',
        'fission(L5.1, after, Init, .a, .b)',
        'fission(L4.1, after, L5.1.a, .a, .b)', 'split(L4.1.b, 2, -1, 0)',
        'split(L5.1.b.b, 2, -1, 0)', 'split(L3.b.b, 2, -1, 0)',
        'reorder(L3.b.b.0, L4.1.b.0, L5.1.b.b.0, L3.b.b.1, L4.1.b.1, L5.1.b.b.1)',
        'merge(L4.0.0, L5.0.0)', 'parallelize(merged.L4.0.0.L5.0.0, openmp)'
    ]
    sch_log = sch.logs()
    print(sch_log)
    assert len(sch_log) == len(std_log)
    for l, r in zip(sch_log, std_log):
        if l.startswith('cache'):
            assert r.startswith('cache')
        else:
            assert l == r
