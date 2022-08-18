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
        #! label: L1
        for i in range(m):
            #! label: L2
            for j in range(m):
                #! label: L4
                for p in range(a):
                    #! label: L5
                    for q in range(a):
                        #! label: Init
                        y[0, 0, p, q] = 0
                        #! label: L3
                        for k in range(b):
                            y[0, 0, p,
                              q] = y[0, 0, p, q] + w[i, j, p, k] * x[i, j, k, q]
                #! label: L6
                for p in range(a):
                    #! label: L7
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
        'split(L4, 4, -1, 0)', 'split(split.outer {L4}, 2, -1, 0)',
        'split(L5, 4, -1, 0)', 'split(split.outer {L5}, 2, -1, 0)',
        'reorder(split.outer {split.outer {L4}}, split.outer {split.outer {L5}}, split.inner {split.outer {L4}}, split.inner {split.outer {L5}}, split.inner {L4}, split.inner {L5})',
        'fission(split.inner {L5}, after, Init)',
        'fission(split.inner {L4}, after, fission.0 {split.inner {L5}})',
        'split(fission.1 {split.inner {L4}}, 2, -1, 0)',
        'split(fission.1 {fission.1 {split.inner {L5}}}, 2, -1, 0)',
        'split(fission.1 {fission.1 {L3}}, 2, -1, 0)',
        'reorder(split.outer {fission.1 {fission.1 {L3}}}, split.outer {fission.1 {split.inner {L4}}}, split.outer {fission.1 {fission.1 {split.inner {L5}}}}, split.inner {fission.1 {fission.1 {L3}}}, split.inner {fission.1 {split.inner {L4}}}, split.inner {fission.1 {fission.1 {split.inner {L5}}}})',
        'merge(split.outer {split.outer {L4}}, split.outer {split.outer {L5}})',
        'parallelize(merge {split.outer {split.outer {L4}}, split.outer {split.outer {L5}}}, openmp)'
    ]
    sch_log = sch.pretty_logs()
    print(sch_log)
    assert sch_log == std_log
