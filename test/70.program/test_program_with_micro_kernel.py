import pytest

import freetensor as ft


@pytest.mark.skipif(not ft.with_pytorch() or not ft.with_cuda(),
                    reason="requires PyTorch and CUDA")
@pytest.mark.parametrize('dtype', ['float16', 'float64'])
def test_matmul(dtype):

    M = N = K = 5000
    block_n = block_m = 128
    block_k = 32
    n_warps = 4

    device = ft.GPU()
    target = device.target()
    with target:

        @ft.transform
        def matmul(a: ft.Var[(M, K), dtype], b: ft.Var[(K, N), dtype]):
            c = ft.empty((M, N), dtype)
            #! label: blk_m
            for i in range(0, M, block_m):
                #! label: blk_n
                for j in range(0, N, block_n):
                    #! label: aa
                    aa = ft.empty((block_m, block_k), dtype)
                    #! label: bb
                    bb = ft.empty((block_k, block_n), dtype)
                    #! label: cc
                    cc = ft.empty((block_m, block_n), dtype)
                    #! label: zero_cc
                    for ii in range(block_m):
                        for jj in range(block_n):
                            cc[ii, jj] = 0
                    for k in range(0, K, block_k):
                        #! label: load_aa
                        for ii in range(block_m):
                            for kk in range(block_k):
                                if i + ii < M and k + kk < K:
                                    aa[ii, kk] = a[i + ii, k + kk]
                                else:
                                    aa[ii, kk] = 0
                        #! label: load_bb
                        for kk in range(block_k):
                            for jj in range(block_n):
                                if k + kk < K and j + jj < N:
                                    bb[kk, jj] = b[k + kk, j + jj]
                                else:
                                    bb[kk, jj] = 0
                        #! label: micro_kernel
                        for ii in range(block_m):
                            for jj in range(block_n):
                                for kk in range(block_k):
                                    cc[ii, jj] += aa[ii, kk] * bb[kk, jj]
                    #! label: flush_cc
                    for ii in range(block_m):
                        for jj in range(block_n):
                            # TODO: Can we avoid using `unbound`?
                            if ft.unbound(i + ii < M and j + jj < N):
                                c[i + ii, j + jj] = cc[ii, jj]
            return c

        s = ft.Schedule(matmul, verbose=2)
        s.parallelize("blk_m", "blockIdx.y")
        s.parallelize("blk_n", "blockIdx.x")
        s.as_matmul("micro_kernel",
                    target=target,
                    backend="cutlass-micro-block",
                    mode=ft.AsMatMulMode.TryVarReorder)
        load_aa_warp, load_aa_thr = s.split(
            s.split(s.merge("load_aa", "<For><-load_aa"), n_warps * 32)[1], 32)
        s.parallelize(load_aa_warp, "threadIdx.y")
        s.parallelize(load_aa_thr, "threadIdx.x")
        load_bb_warp, load_bb_thr = s.split(
            s.split(s.merge("load_bb", "<For><-load_bb"), n_warps * 32)[1], 32)
        s.parallelize(load_bb_warp, "threadIdx.y")
        s.parallelize(load_bb_thr, "threadIdx.x")
        s.parallelize_as("zero_cc", "$as_matmul{micro_kernel}", "cc")
        s.parallelize_as("flush_cc", "$as_matmul{micro_kernel}", "cc")
        s.set_mem_type("aa", "gpu/shared")
        s.set_mem_type("bb", "gpu/shared")
        s.set_mem_type("cc", "gpu/local")
        scheduled = s.func()
        exe = ft.optimize(scheduled, verbose=2)

        import torch

        dtype_to_torch = {
            'float16': torch.float16,
            'float64': torch.float64,
        }
        a_torch = torch.rand(M, K, dtype=dtype_to_torch[dtype]).cuda()
        b_torch = torch.rand(K, N, dtype=dtype_to_torch[dtype]).cuda()
        y_std = a_torch @ b_torch
        a_arr = ft.array(a_torch)
        b_arr = ft.array(b_torch)
        y_arr = exe(a_arr, b_arr)
        y_torch = y_arr.torch()

        if dtype == 'float16':
            assert torch.all(torch.isclose(y_torch, y_std, rtol=2e-2))
        else:
            assert torch.all(torch.isclose(y_torch, y_std))
