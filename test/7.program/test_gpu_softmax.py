import torch
import numpy as np

import ir
import ir.debug
import ir.libop
from ir.libop import StaticType as T

target = ir.GPU()
device = ir.Device(target)


def test_manual_static():
    # softmax in BERT
    # shape = batch_size * #heads * seq_len * seq_len

    batch_size = 1
    n_heads = 16
    seq_len = 512
    seq_len = 512

    @ir.transform
    def f(x, y):
        "nid: V_x"
        ir.declare_var(x, (batch_size, n_heads, seq_len, seq_len),
                       "float32",
                       "input",
                       "gpu/global",
                       name="x")
        "nid: V_y"
        ir.declare_var(y, (batch_size, n_heads, seq_len, seq_len),
                       "float32",
                       "output",
                       "gpu/global",
                       name="y")
        "nid: softmax"
        ir.libop.softmax_(T("float32", 4), T("float32", 4),
                          "gpu/global")([batch_size, n_heads, seq_len, seq_len],
                                        [batch_size, n_heads, seq_len, seq_len],
                                        x, y)

    print(f.pretty_print())
    s = ir.Schedule(f)

    # Inline shape inference
    s.inline("softmax:x_shape")
    s.inline("softmax:V_x:V_y_shape")
    s.inline("softmax:max:y_shape")
    s.inline("softmax:sub:broadcast_shape:out_shape")
    s.inline("softmax:sum:y_shape")
    s.inline("softmax:div:broadcast_shape:out_shape")
    s.inline("softmax:y_shape")

    # L_head
    L_head = s.fuse("softmax:max:recur:init:recur:L",
                    "softmax:max:recur:reduce:recur:L")
    L_head = s.fuse(L_head, "softmax:sub:recur:recur:L_elem")
    L_head = s.fuse(L_head, "softmax:exp:recur:recur:L_elem")
    L_head = s.fuse(L_head, "softmax:sum:recur:init:recur:L")
    L_head = s.fuse(L_head, "softmax:sum:recur:reduce:recur:L")
    L_head = s.fuse(L_head, "softmax:div:recur:L_elem")

    # L_seq_outer
    L_seq_outer = s.fuse("softmax:max:recur:init:recur:recur:L",
                         "softmax:max:recur:reduce:recur:recur:L")
    L_seq_outer = s.fuse(L_seq_outer, "softmax:sub:recur:recur:recur:L_elem")
    L_seq_outer = s.fuse(L_seq_outer, "softmax:exp:recur:recur:recur:L_elem")
    L_seq_outer = s.fuse(L_seq_outer, "softmax:sum:recur:init:recur:recur:L")
    L_seq_outer = s.fuse(L_seq_outer, "softmax:sum:recur:reduce:recur:recur:L")
    L_seq_outer = s.fuse(L_seq_outer, "softmax:div:recur:recur:L_elem")

    # ----------------

    # Don't store these intermediates
    s.inline("softmax:sub:out")

    # Store these intermedates to registers
    load_x, _, _, _ = s.cache(
        s.find(lambda x: x.nid() == L_seq_outer).node().body, "x", "gpu/local")
    load_x_loop = s.find(lambda x: x.nid() == load_x).outer()
    s.cache(
        s.find(lambda x: x.nid() == "softmax:exp:y").node().body,
        "$softmax:exp:y", "gpu/local")

    # ----------------

    thr_x_dim = 32
    thr_y_dim = 32

    # Optimize reductions
    def opt_red(def_nid, var, init_nid, loop_nid):
        # Hold result in shared memory
        _, _, V_sum_shmem, _ = s.cache(
            s.find(lambda x: x.nid() == def_nid).node().body, var, "gpu/shared")

        # Parallel reduction
        thr_x, serial = s.split(loop_nid, nparts=thr_x_dim)
        _, red_final, V_sum_partial, _ = s.cache_reduction(
            serial, V_sum_shmem, "gpu/shared")
        red_final = s.move_to(red_final, ir.MoveToSide.After, thr_x)
        init_nid = s.move_to(init_nid, ir.MoveToSide.Before, red_final)
        s.parallelize(thr_x, "threadIdx.x")

        # Reduce partial results in registers
        s.cache_reduction(serial, V_sum_partial, "gpu/local")

        # Reduce global results in registers
        s.cache_reduction(red_final, V_sum_shmem, "gpu/local")

        return V_sum_shmem

    V_sum_shmem = opt_red("softmax:sum:y", "$softmax:sum:y",
                          "softmax:sum:recur:init:recur:recur:recur:recur:exec",
                          "softmax:sum:recur:reduce:recur:recur:recur:L")
    V_max_shmem = opt_red("softmax:max:y", "$softmax:max:y",
                          "softmax:max:recur:init:recur:recur:recur:recur:exec",
                          "softmax:max:recur:reduce:recur:recur:recur:L")

    # Parallelize data-parall loops
    def opt_elemwise(loop_nid):
        thr_x, serial = s.split(loop_nid, nparts=thr_x_dim)
        s.parallelize(thr_x, "threadIdx.x")
        return thr_x, serial

    opt_elemwise(load_x_loop)
    exp_outer, exp_inner = opt_elemwise(
        "softmax:exp:recur:recur:recur:recur:L_elem")
    div_outer, div_inner = opt_elemwise("softmax:div:recur:recur:recur:L_elem")
    s.cache(exp_inner, V_max_shmem, "gpu/local")
    s.cache(div_inner, V_sum_shmem, "gpu/local")

    # Parallelize outer loops
    L_indep = s.merge(L_head, L_seq_outer)
    L_blk, L_thr_y = s.split(L_indep, thr_y_dim)
    s.parallelize(L_blk, "blockIdx.x")
    s.parallelize(L_thr_y, "threadIdx.y")

    f = ir.lower(s.func(), target)
    print(f.pretty_print())

    code = ir.codegen(f, target)
    print(ir.debug.with_line_no(code))

    x_torch = torch.rand(batch_size,
                         n_heads,
                         seq_len,
                         seq_len,
                         dtype=torch.float32)
    x_arr = ir.Array(x_torch.numpy(), device)
    y_torch = torch.zeros(batch_size,
                          n_heads,
                          seq_len,
                          seq_len,
                          dtype=torch.float32)
    y_arr = ir.Array(y_torch.numpy(), device)
    ir.Driver(f, code, device)(x_arr, y_arr)
    y_torch = torch.Tensor(y_arr.numpy().reshape(batch_size, n_heads, seq_len,
                                                 seq_len))

    assert torch.all(torch.isclose(y_torch, torch.softmax(x_torch, axis=-1)))
