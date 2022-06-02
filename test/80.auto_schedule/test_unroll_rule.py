import freetensor as ft
import numpy as np
import pytest

target = ft.GPU()
device = ft.Device(target)


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
    s = ft.AutoSchedule(s, target, device, 8)
    sch = s.test_unroll()
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
        'split(L4, factor=2, nparts=-1, shift=0)',
        'split(L4.0, factor=2, nparts=-1, shift=0)',
        'split(L4.0.0, factor=2, nparts=-1, shift=0)',
        'split(L4.0.0.0, factor=2, nparts=-1, shift=0)',
        'split(L5, factor=2, nparts=-1, shift=0)',
        'split(L5.0, factor=2, nparts=-1, shift=0)',
        'split(L5.0.0, factor=2, nparts=-1, shift=0)',
        'split(L5.0.0.0, factor=2, nparts=-1, shift=0)',
        'split(L3, factor=2, nparts=-1, shift=0)',
        'split(L3.0, factor=2, nparts=-1, shift=0)',
        'reorder(L4.0.0.0.0, L5.0.0.0.0, L4.0.0.0.1, L5.0.0.0.1, L4.0.0.1, L5.0.0.1, L3.0.0, L3.0.1, L4.0.1, L5.0.1, L3.1, L4.1, L5.1)',
        'split(L6, factor=4, nparts=-1, shift=0)',
        'split(L6.0, factor=2, nparts=-1, shift=0)',
        'split(L6.0.0, factor=2, nparts=-1, shift=0)',
        'split(L7, factor=4, nparts=-1, shift=0)',
        'split(L7.0, factor=2, nparts=-1, shift=0)',
        'split(L7.0.0, factor=2, nparts=-1, shift=0)',
        'reorder(L6.0.0.0, L7.0.0.0, L6.0.0.1, L7.0.0.1, L6.0.1, L7.0.1, L6.1, L7.1)',
        'fuse(L4.0.0.0.0, L6.0.0.0)', 'fuse(L5.0.0.0.0, L7.0.0.0)',
        'fuse(L4.0.0.0.1, L6.0.0.1)', 'fuse(L5.0.0.0.1, L7.0.0.1)',
        'fuse(L4.0.0.1, L6.0.1)', 'fuse(L5.0.0.1, L7.0.1)', 'cache(#34, y)',
        'cache(L3.0.1, w)', 'cache(#58, x)',
        'merge(fused.L4.0.0.0.0.L6.0.0.0, fused.L5.0.0.0.0.L7.0.0.0)',
        'merge(fused.L4.0.0.0.1.L6.0.0.1, fused.L5.0.0.0.1.L7.0.0.1)',
        'merge(fused.L4.0.0.1.L6.0.1, fused.L5.0.0.1.L7.0.1)',
        'parallelize(merged.fused.L4.0.0.0.0.L6.0.0.0.fused.L5.0.0.0.0.L7.0.0.0, blockIdx.x)',
        'blend(merged.fused.L4.0.0.0.1.L6.0.0.1.fused.L5.0.0.0.1.L7.0.0.1)',
        'parallelize(merged.fused.L4.0.0.1.L6.0.1.fused.L5.0.0.1.L7.0.1, threadIdx.x)',
        'unroll(#37)', 'unroll(#81)', 'unroll(#60)', 'unroll(L5.1)',
        'unroll(L4.1)', 'unroll(L7.1)', 'unroll(#43)'
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
