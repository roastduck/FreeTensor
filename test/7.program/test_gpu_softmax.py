import torch
import numpy as np

import ir
import ir.debug
import ir.libop

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
        ir.declare_var(x, (batch_size, n_heads, seq_len, seq_len), "float32",
                       "input", "gpu/global")
        "nid: V_y"
        ir.declare_var(y, (batch_size, n_heads, seq_len, seq_len), "float32",
                       "output", "gpu/global")
        "nid: softmax"
        ir.libop.softmax_(x, y)

    print(f)
    s = ir.Schedule(f)

    # L_head
    L_head = s.fuse("softmax->max->impl->recur->init->recur->L",
                    "softmax->max->impl->recur->reduce->recur->L")
    L_head = s.fuse(L_head, "softmax->sub->recur->recur->L_elem")
    L_head = s.fuse(L_head, "softmax->exp->recur->recur->L_elem")
    L_head = s.fuse(L_head, "softmax->sum->recur->init->recur->L")
    L_head = s.fuse(L_head, "softmax->sum->recur->reduce->recur->L")
    L_head = s.fuse(L_head, "softmax->div->recur->L_elem")

    # L_seq_outer
    L_seq_outer = s.fuse("softmax->max->impl->recur->init->recur->recur->L",
                         "softmax->max->impl->recur->reduce->recur->recur->L")
    L_seq_outer = s.fuse(L_seq_outer,
                         "softmax->sub->recur->recur->recur->L_elem")
    L_seq_outer = s.fuse(L_seq_outer,
                         "softmax->exp->recur->recur->recur->L_elem")
    L_seq_outer = s.fuse(L_seq_outer,
                         "softmax->sum->recur->init->recur->recur->L")
    L_seq_outer = s.fuse(L_seq_outer,
                         "softmax->sum->recur->reduce->recur->recur->L")
    L_seq_outer = s.fuse(L_seq_outer, "softmax->div->recur->recur->L_elem")

    # ----------------

    # Don't store these intermediates
    s.inline("softmax->sub->out")

    # Store these intermedates to registers
    load_x, _, _, x_local_def = s.cache(
        s.find(L_seq_outer).body, "x", "gpu/global")
    load_x_loop = s.find(load_x).parent_stmt()

    # ----------------

    thr_x_dim = 32
    thr_y_dim = 32

    # Optimize reductions
    def opt_red(def_nid, init_nid, loop_nid):
        node = s.find(def_nid)

        # Hold result in shared memory
        _, _, V_sum_shmem, _ = s.cache(node.body, node.name, "gpu/shared")

        # Parallel reduction
        serial, thr_x = s.split(loop_nid, thr_x_dim)
        s.reorder([thr_x, serial])
        s.parallelize(thr_x, "threadIdx.x")

        # Reduce partial results in registers
        s.cache_reduction(serial, V_sum_shmem, "gpu/local")

        return V_sum_shmem

    V_sum_shmem = opt_red(
        "softmax->sum->y",
        "softmax->sum->recur->init->recur->recur->recur->recur->exec",
        "softmax->sum->recur->reduce->recur->recur->recur->L")
    V_max_shmem = opt_red(
        "softmax->max->impl->y",
        "softmax->max->impl->recur->init->recur->recur->recur->recur->exec",
        "softmax->max->impl->recur->reduce->recur->recur->recur->L")

    # Parallelize data-parall loops
    def opt_elemwise(loop_nid):
        serial, thr_x = s.split(loop_nid, thr_x_dim)
        s.reorder([thr_x, serial])
        s.parallelize(thr_x, "threadIdx.x")
        return thr_x, serial

    opt_elemwise(load_x_loop)
    exp_outer, exp_inner = opt_elemwise(
        "softmax->exp->recur->recur->recur->recur->L_elem")
    div_outer, div_inner = opt_elemwise(
        "softmax->div->recur->recur->recur->L_elem")
    s.cache(exp_inner, V_max_shmem, "gpu/local")
    s.cache(div_inner, V_sum_shmem, "gpu/local")

    # Parallelize outer loops
    L_indep = s.merge(L_head, L_seq_outer)
    L_blk, L_thr_y = s.split(L_indep, thr_y_dim)
    s.parallelize(L_blk, "blockIdx.x")
    s.parallelize(L_thr_y, "threadIdx.y")

    # ----------------

    s.set_mem_type(x_local_def, "gpu/local")
    s.set_mem_type(s.find("softmax->exp->y"), "gpu/local")

    f = ir.lower(s.func(), target)
    print(f)

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
    y_torch = torch.Tensor(y_arr.numpy())

    assert torch.all(torch.isclose(y_torch, torch.softmax(x_torch, axis=-1)))
