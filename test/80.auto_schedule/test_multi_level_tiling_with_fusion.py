import freetensor as ft
import numpy as np

target = ft.CPU()
device = ft.Device(target.type())


def test_fusion():
    a = 128
    b = 256
    m = 4

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
                #! label: L3
                for k in range(b):
                    #! label: L4
                    for p in range(a):
                        #! label: L5
                        for q in range(a):
                            y[0, 0, p,
                              q] = y[0, 0, p, q] + w[i, j, p, k] * x[i, j, k, q]
                #! label: L6
                for p in range(a):
                    #! label: L7
                    for q in range(a):
                        z[i, j, p, q] = y[0, 0, p, q]

    s = ft.Schedule(test)
    print(s.ast())
    s = ft.AutoSchedule(s,
                        target,
                        device,
                        rule_set={"multi_level_tiling_with_fusion"})
    sch = s.test_round({"multi_level_tiling_with_fusion": 1})
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
        'split(L4, 2, -1, 0)', 'split(split.outer {L4}, 2, -1, 0)',
        'split(split.outer {split.outer {L4}}, 2, -1, 0)',
        'split(L5, 2, -1, 0)', 'split(split.outer {L5}, 2, -1, 0)',
        'split(split.outer {split.outer {L5}}, 2, -1, 0)',
        'split(L3, 2, -1, 0)',
        'reorder(split.outer {split.outer {split.outer {L4}}}, split.outer {split.outer {split.outer {L5}}}, split.inner {split.outer {split.outer {L4}}}, split.inner {split.outer {split.outer {L5}}}, split.outer {L3}, split.inner {split.outer {L4}}, split.inner {split.outer {L5}}, split.inner {L3}, split.inner {L4}, split.inner {L5})',
        'split(L6, 4, -1, 0)', 'split(split.outer {L6}, 2, -1, 0)',
        'split(L7, 4, -1, 0)', 'split(split.outer {L7}, 2, -1, 0)',
        'reorder(split.outer {split.outer {L6}}, split.outer {split.outer {L7}}, split.inner {split.outer {L6}}, split.inner {split.outer {L7}}, split.inner {L6}, split.inner {L7})',
        'fuse(split.outer {split.outer {split.outer {L4}}}, split.outer {split.outer {L6}}, false)',
        'fuse(split.outer {split.outer {split.outer {L5}}}, split.outer {split.outer {L7}}, false)',
        'fuse(split.inner {split.outer {split.outer {L4}}}, split.inner {split.outer {L6}}, false)',
        'fuse(split.inner {split.outer {split.outer {L5}}}, split.inner {split.outer {L7}}, false)',
        'cache(*, y, cpu)'
    ]
    sch_log = sch.pretty_logs()
    print(sch_log)
    assert std_log[:-1] == sch_log[:-1]
    assert sch_log[-1].startswith('cache(') and sch_log[-1].endswith(
        ', y, cpu)')
