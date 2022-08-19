import freetensor as ft
import numpy as np
import pytest

if not ft.with_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

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
        'split(L4, 2, -1, 0)', 'split($split.0{L4}, 2, -1, 0)',
        'split($split.0{$split.0{L4}}, 2, -1, 0)',
        'split($split.0{$split.0{$split.0{L4}}}, 2, -1, 0)',
        'split(L5, 2, -1, 0)', 'split($split.0{L5}, 2, -1, 0)',
        'split($split.0{$split.0{L5}}, 2, -1, 0)',
        'split($split.0{$split.0{$split.0{L5}}}, 2, -1, 0)',
        'split(L3, 2, -1, 0)', 'split($split.0{L3}, 2, -1, 0)',
        'reorder($split.0{$split.0{$split.0{$split.0{L4}}}}, $split.0{$split.0{$split.0{$split.0{L5}}}}, $split.1{$split.0{$split.0{$split.0{L4}}}}, $split.1{$split.0{$split.0{$split.0{L5}}}}, $split.1{$split.0{$split.0{L4}}}, $split.1{$split.0{$split.0{L5}}}, $split.0{$split.0{L3}}, $split.1{$split.0{L3}}, $split.1{$split.0{L4}}, $split.1{$split.0{L5}}, $split.1{L3}, $split.1{L4}, $split.1{L5})',
        'split(L6, 4, -1, 0)', 'split($split.0{L6}, 2, -1, 0)',
        'split($split.0{$split.0{L6}}, 2, -1, 0)', 'split(L7, 4, -1, 0)',
        'split($split.0{L7}, 2, -1, 0)',
        'split($split.0{$split.0{L7}}, 2, -1, 0)',
        'reorder($split.0{$split.0{$split.0{L6}}}, $split.0{$split.0{$split.0{L7}}}, $split.1{$split.0{$split.0{L6}}}, $split.1{$split.0{$split.0{L7}}}, $split.1{$split.0{L6}}, $split.1{$split.0{L7}}, $split.1{L6}, $split.1{L7})',
        'fuse($split.0{$split.0{$split.0{$split.0{L4}}}}, $split.0{$split.0{$split.0{L6}}}, false)',
        'fuse($split.0{$split.0{$split.0{$split.0{L5}}}}, $split.0{$split.0{$split.0{L7}}}, false)',
        'fuse($split.1{$split.0{$split.0{$split.0{L4}}}}, $split.1{$split.0{$split.0{L6}}}, false)',
        'fuse($split.1{$split.0{$split.0{$split.0{L5}}}}, $split.1{$split.0{$split.0{L7}}}, false)',
        'fuse($split.1{$split.0{$split.0{L4}}}, $split.1{$split.0{L6}}, false)',
        'fuse($split.1{$split.0{$split.0{L5}}}, $split.1{$split.0{L7}}, false)',
        'cache(#34, y, ?)',
        'merge($fuse{$split.0{$split.0{$split.0{$split.0{L4}}}}, $split.0{$split.0{$split.0{L6}}}}, $fuse{$split.0{$split.0{$split.0{$split.0{L5}}}}, $split.0{$split.0{$split.0{L7}}}})',
        'merge($fuse{$split.1{$split.0{$split.0{$split.0{L4}}}}, $split.1{$split.0{$split.0{L6}}}}, $fuse{$split.1{$split.0{$split.0{$split.0{L5}}}}, $split.1{$split.0{$split.0{L7}}}})',
        'merge($fuse{$split.1{$split.0{$split.0{L4}}}, $split.1{$split.0{L6}}}, $fuse{$split.1{$split.0{$split.0{L5}}}, $split.1{$split.0{L7}}})',
        'parallelize($merge{$fuse{$split.0{$split.0{$split.0{$split.0{L4}}}}, $split.0{$split.0{$split.0{L6}}}}, $fuse{$split.0{$split.0{$split.0{$split.0{L5}}}}, $split.0{$split.0{$split.0{L7}}}}}, blockIdx.x)',
        'blend($merge{$fuse{$split.1{$split.0{$split.0{$split.0{L4}}}}, $split.1{$split.0{$split.0{L6}}}}, $fuse{$split.1{$split.0{$split.0{$split.0{L5}}}}, $split.1{$split.0{$split.0{L7}}}}})',
        'parallelize($merge{$fuse{$split.1{$split.0{$split.0{L4}}}, $split.1{$split.0{L6}}}, $fuse{$split.1{$split.0{$split.0{L5}}}, $split.1{$split.0{L7}}}}, threadIdx.x)',
        'unroll(#37, false)', 'unroll($split.1{L5}, false)',
        'unroll($split.1{L4}, false)', 'unroll($split.1{L7}, false)',
        'unroll(#43, false)'
    ]
    sch_log = sch.pretty_logs()
    print(sch_log)
    assert len(sch_log) == len(std_log)
    for l, r in zip(sch_log, std_log):
        if l.startswith('cache(#'):
            assert r.startswith('cache(#')
        elif l.startswith('unroll(#'):
            assert r.startswith('unroll(#')
        else:
            assert l == r
