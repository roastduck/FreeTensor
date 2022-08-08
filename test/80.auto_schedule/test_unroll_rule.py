import freetensor as ft
import numpy as np
import pytest

device = ft.GPU()
target = device.target()


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_unroll():
    a = 128
    b = 128
    m = 4

    @ft.transform
    def test(w, x, y, z):
        w: ft.Var[(m, m, a, b), "int32", "input", "gpu/global"]
        x: ft.Var[(m, m, b, a), "int32", "input", "gpu/global"]
        y: ft.Var[(1, 1, a, a), "int32", "inout", "gpu/global"]
        z: ft.Var[(m, m, a, a), "int32", "output", "gpu/global"]
        #! nid: L1
        for i in range(m):
            #! nid: L2
            for j in range(m):
                #! nid: L3
                for k in range(b):
                    #! nid: L4
                    for p in range(a):
                        #! nid: L5
                        for q in range(a):
                            y[0, 0, p,
                              q] = y[0, 0, p, q] + w[i, j, p, k] * x[i, j, k, q]
                #! nid: L6
                for p in range(a):
                    #! nid: L7
                    for q in range(a):
                        z[i, j, p, q] = y[0, 0, p, q]

    s = ft.Schedule(test)
    print(s.ast())
    s = ft.AutoSchedule(
        s,
        target,
        device,
        rule_set={"multi_level_tiling_with_fusion", "thread_bind", "unroll"})
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
        'split(L4, 2, -1, 0)', 'split(L4.0, 2, -1, 0)',
        'split(L4.0.0, 2, -1, 0)', 'split(L4.0.0.0, 2, -1, 0)',
        'split(L5, 2, -1, 0)', 'split(L5.0, 2, -1, 0)',
        'split(L5.0.0, 2, -1, 0)', 'split(L5.0.0.0, 2, -1, 0)',
        'split(L3, 2, -1, 0)', 'split(L3.0, 2, -1, 0)',
        'reorder(L4.0.0.0.0, L5.0.0.0.0, L4.0.0.0.1, L5.0.0.0.1, L4.0.0.1, L5.0.0.1, L3.0.0, L3.0.1, L4.0.1, L5.0.1, L3.1, L4.1, L5.1)',
        'split(L6, 4, -1, 0)', 'split(L6.0, 2, -1, 0)',
        'split(L6.0.0, 2, -1, 0)', 'split(L7, 4, -1, 0)',
        'split(L7.0, 2, -1, 0)', 'split(L7.0.0, 2, -1, 0)',
        'reorder(L6.0.0.0, L7.0.0.0, L6.0.0.1, L7.0.0.1, L6.0.1, L7.0.1, L6.1, L7.1)',
        'fuse(L4.0.0.0.0, L6.0.0.0, false)',
        'fuse(L5.0.0.0.0, L7.0.0.0, false)',
        'fuse(L4.0.0.0.1, L6.0.0.1, false)',
        'fuse(L5.0.0.0.1, L7.0.0.1, false)', 'fuse(L4.0.0.1, L6.0.1, false)',
        'fuse(L5.0.0.1, L7.0.1, false)', 'cache(#34, y, ?)',
        'merge(fused.L4.0.0.0.0.L6.0.0.0, fused.L5.0.0.0.0.L7.0.0.0)',
        'merge(fused.L4.0.0.0.1.L6.0.0.1, fused.L5.0.0.0.1.L7.0.0.1)',
        'merge(fused.L4.0.0.1.L6.0.1, fused.L5.0.0.1.L7.0.1)',
        'parallelize(merged.fused.L4.0.0.0.0.L6.0.0.0.fused.L5.0.0.0.0.L7.0.0.0, blockIdx.x)',
        'blend(merged.fused.L4.0.0.0.1.L6.0.0.1.fused.L5.0.0.0.1.L7.0.0.1)',
        'parallelize(merged.fused.L4.0.0.1.L6.0.1.fused.L5.0.0.1.L7.0.1, threadIdx.x)',
        'unroll(#37, false)', 'unroll(L5.1, false)', 'unroll(L4.1, false)',
        'unroll(L7.1, false)', 'unroll(#43, false)'
    ]
    sch_log = sch.logs()
    print(sch_log)
    assert len(sch_log) == len(std_log)
    for l, r in zip(sch_log, std_log):
        if l.startswith('cache(#'):
            assert r.startswith('cache(#')
        elif l.startswith('unroll(#'):
            assert r.startswith('unroll(#')
        else:
            assert l == r
