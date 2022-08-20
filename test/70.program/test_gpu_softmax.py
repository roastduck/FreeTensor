import torch
import pytest
import numpy as np

import freetensor as ft
from freetensor import debug
from freetensor import libop

if not ft.with_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

device = ft.GPU()
target = device.target()


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_manual_static():
    # softmax in BERT
    # shape = batch_size * #heads * seq_len * seq_len

    batch_size = 1
    n_heads = 16
    seq_len = 512
    seq_len = 512

    @ft.transform
    def f(x, y):
        #! label: V_x
        x: ft.Var[(batch_size, n_heads, seq_len, seq_len), "float32", "input",
                  "gpu/global"]
        #! label: V_y
        y: ft.Var[(batch_size, n_heads, seq_len, seq_len), "float32", "output",
                  "gpu/global"]
        #! label: softmax
        libop.softmax_(x, y)

    print(f)
    s = ft.Schedule(f)

    # L_head
    L_head = s.fuse("L<~recur<~init<~recur<~impl<~max<~softmax")
    L_head = s.fuse(L_head)
    L_head = s.fuse(L_head)
    L_head = s.fuse(L_head)
    L_head = s.fuse(L_head)

    # L_seq_outer
    L_seq_outer = s.fuse("L<~recur<~recur<~init<~recur<~impl<~max<~softmax")
    L_seq_outer = s.fuse(L_seq_outer)
    L_seq_outer = s.fuse(L_seq_outer)
    L_seq_outer = s.fuse(L_seq_outer)
    L_seq_outer = s.fuse(L_seq_outer)

    # ----------------

    # Don't store these intermediates
    s.inline("out<~sub<~softmax")

    # Store these intermedates to registers
    load_x, _, _, x_local_def = s.cache(
        s.find(L_seq_outer).body, "x", "gpu/global")
    load_x_loop = s.find(load_x).parent_stmt(
        lambda s: s.type() == ft.ASTNodeType.For)

    # ----------------

    thr_x_dim = 32
    thr_y_dim = 32

    # Optimize reductions
    def opt_red(def_label, init_label, loop_label):
        node = s.find(def_label)

        # Hold result in shared memory
        _, _, V_sum_shmem, _ = s.cache(node.body, node.name, "gpu/shared")

        # Parallel reduction
        serial, thr_x = s.split(loop_label, thr_x_dim)
        s.reorder([thr_x, serial])
        s.parallelize(thr_x, "threadIdx.x")

        # Reduce partial results in registers
        s.cache_reduction(serial, V_sum_shmem, "gpu/local")

        return V_sum_shmem

    V_sum_shmem = opt_red(
        "y<~sum<~softmax",
        "exec<~recur<~recur<~recur<~recur<~init<~recur<~sum<~softmax",
        "L<~recur<~recur<~recur<~reduce<~recur<~sum<~softmax")
    V_max_shmem = opt_red(
        "y<~impl<~max<~softmax",
        "exec<~recur<~recur<~recur<~recur<~init<~recur<~impl<~max<~softmax",
        "L<~recur<~recur<~recur<~reduce<~recur<~impl<~max<~softmax")

    # Parallelize data-parall loops
    def opt_elemwise(loop_label):
        serial, thr_x = s.split(loop_label, thr_x_dim)
        s.reorder([thr_x, serial])
        s.parallelize(thr_x, "threadIdx.x")
        return thr_x, serial

    opt_elemwise(load_x_loop)
    exp_outer, exp_inner = opt_elemwise(
        "L_elem<~recur<~recur<~recur<~recur<~exp<~softmax")
    div_outer, div_inner = opt_elemwise(
        "L_elem<~recur<~recur<~recur<~div<~softmax")
    s.cache(exp_inner, V_max_shmem, "gpu/local")
    s.cache(div_inner, V_sum_shmem, "gpu/local")

    # Parallelize outer loops
    L_indep = s.merge(L_head, L_seq_outer)
    L_blk, L_thr_y = s.split(L_indep, thr_y_dim)
    s.parallelize(L_blk, "blockIdx.x")
    s.parallelize(L_thr_y, "threadIdx.y")

    # ----------------

    s.set_mem_type(x_local_def, "gpu/local")
    s.set_mem_type(s.find("y<~exp<~softmax"), "gpu/local")

    f = ft.lower(s.func(), target)
    print(f)

    code = ft.codegen(f, target)
    print(debug.with_line_no(code))

    x_torch = torch.rand(batch_size,
                         n_heads,
                         seq_len,
                         seq_len,
                         dtype=torch.float32)
    x_arr = ft.Array(x_torch.numpy())
    y_torch = torch.zeros(batch_size,
                          n_heads,
                          seq_len,
                          seq_len,
                          dtype=torch.float32)
    y_arr = ft.Array(y_torch.numpy())
    ft.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch, torch.softmax(x_torch, axis=-1)))
