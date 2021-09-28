import sys
import time
import itertools
import numpy as np
import ir
import ir.debug


def load_data(data_name: str):
    '''
    Load data from ../data/{data_name}.config and ../data/{data_name}.graph

    Returns (#vertices, #edges, ptr, idx), where ptr and idx forms a CSR format
    '''

    with open(f"../data/{data_name}.config", 'r') as f:
        num_v, num_e = map(int, f.readline().strip().split(' '))

    with open(f"../data/{data_name}.graph", 'r') as f:
        ptr = np.array(list(map(int, f.readline().strip().split(" "))))
        idx = np.array(list(map(int, f.readline().strip().split(" "))))

    return num_v, num_e, ptr, idx


jit_cache = {}


def gat_layer(ptr, idx, feat, weight, attn_l, attn_r, y, num_v, num_e, feat_len,
              device, mtype, shared_mtype, local_mtype):
    if (num_v, num_e, feat_len) in jit_cache:
        exe = jit_cache[(num_v, num_e, feat_len)]

    else:

        inf = float("inf")

        @ir.transform
        def f(ptr, idx, feat, weight, attn_l, attn_r, y):
            ir.declare_var(ptr, (num_v + 1,), "int32", "input", mtype)
            ir.declare_var(idx, (num_e,), "int32", "input", mtype)
            ir.declare_var(feat, (num_v, feat_len), "float32", "input", mtype)
            ir.declare_var(weight, (feat_len, feat_len), "float32", "input",
                           mtype)
            ir.declare_var(attn_l, (feat_len,), "float32", "input", mtype)
            ir.declare_var(attn_r, (feat_len,), "float32", "input", mtype)
            ir.declare_var(y, (num_v, feat_len), "float32", "output", mtype)

            feat2 = ir.create_var((num_v, feat_len), "float32", "cache", mtype)
            'nid: L_feat2'
            for i in range(num_v):
                for j in range(feat_len):
                    feat2[i, j] = 0
                    for k in range(feat_len):
                        feat2[i, j] += feat[i, k] * weight[k, j]

            att_l = ir.create_var((num_v,), "float32", "cache", mtype)
            'nid: L_att_l'
            for i in range(num_v):
                att_l[i] = 0
                for j in range(feat_len):
                    att_l[i] += feat2[i, j] * attn_l[j]

            att_r = ir.create_var((num_v,), "float32", "cache", mtype)
            'nid: L_att_r'
            for i in range(num_v):
                att_r[i] = 0
                for j in range(feat_len):
                    att_r[i] += feat2[i, j] * attn_r[j]

            edge_exp = ir.create_var((num_e,), "float32", "cache", mtype)
            edge_norm = ir.create_var((num_e,), "float32", "cache", mtype)
            'nid: Li'
            'no_deps'
            for i in range(num_v):
                edge_max = ir.create_var((), "float32", "cache", shared_mtype)
                edge_max[()] = -inf
                'nid: Lk1'
                for k in range(ptr[i], ptr[i + 1]):
                    e = ir.create_var((), "float32", "cache", local_mtype)
                    e[()] = att_l[idx[k]] + att_r[i]
                    edge_exp[k] = ir.exp(
                        ir.if_then_else(e[()] >= 0, e[()], e[()] * 0.1))
                    edge_max[()] = ir.max(edge_max[()], edge_exp[k])
                edge_sum = ir.create_var((), "float32", "cache", shared_mtype)
                edge_sum[()] = 0
                'nid: Lk2'
                for k in range(ptr[i], ptr[i + 1]):
                    edge_norm[k] = edge_exp[k] - edge_max[()]
                    edge_sum[()] += edge_norm[k]
                'nid: Lj'
                for j in range(feat_len):
                    y[i, j] = 0
                    'nid: Lk3'
                    for k in range(ptr[i], ptr[i + 1]):
                        y[i,
                          j] += feat2[idx[k], j] * edge_norm[k] / edge_sum[()]

        s = ir.Schedule(f)
        if device.target().type() == ir.TargetType.CPU:
            s.as_matmul('L_feat2')
            s.as_matmul('L_att_l')
            s.as_matmul('L_att_r')
            s.parallelize('Li', 'openmp')
        else:
            s.as_matmul('L_feat2')
            s.as_matmul('L_att_l')
            s.as_matmul('L_att_r')

            blk, thr_y = s.split('Li', 4)
            s.parallelize(blk, 'blockIdx.x')
            s.parallelize(thr_y, 'threadIdx.y')

            inner, outer = s.split('Lk1', 32)
            s.reorder([outer, inner])
            init, final, red_shared, _ = s.cache_reduction(
                inner, "$:edge_max", "gpu/shared")
            final = s.move_to(final, ir.MoveToSide.After, outer)
            s.cache_reduction(inner, red_shared, "gpu/local")
            s.cache_reduction(final, '$:edge_max', 'gpu/local')
            s.parallelize(outer, 'threadIdx.x')

            s.cache('Lk3', '$:edge_max', 'gpu/local')
            inner, outer = s.split('Lk2', 32)
            s.reorder([outer, inner])
            init, final, red_shared, _ = s.cache_reduction(
                inner, "$:edge_sum", "gpu/shared")
            final = s.move_to(final, ir.MoveToSide.After, outer)
            s.cache_reduction(inner, red_shared, "gpu/local")
            s.cache_reduction(final, '$:edge_sum', 'gpu/local')
            s.parallelize(outer, 'threadIdx.x')

            s.parallelize('Lj', 'threadIdx.x')
            s.cache('Lk3', 'y', 'gpu/local')
            s.cache('Lk3', '$:edge_sum', 'gpu/local')
        f = ir.lower(s.func(), device.target())
        print(f)
        code = ir.codegen(f, device.target())
        print(ir.debug.with_line_no(code))
        exe = ir.Driver(f, code, device)
        exe.set_params(ptr, idx, feat, weight, attn_l, attn_r, y)
        # TODO: do not set_params here
        jit_cache[(num_v, num_e, feat_len)] = exe

    exe.run()
    exe.sync()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <cpu/gpu> <data_name>")
        exit(-1)
    device = sys.argv[1]
    data_name = sys.argv[2]

    num_v, num_e, ptr, idx = load_data(data_name)

    feat_len = 32
    ptr = ptr.astype("int32")
    idx = idx.astype("int32")
    x = np.random.uniform(size=(num_v, feat_len)).astype("float32")
    w = np.random.uniform(size=(feat_len, feat_len)).astype("float32")
    w_attn_1 = np.random.uniform(size=(feat_len,)).astype("float32")
    w_attn_2 = np.random.uniform(size=(feat_len,)).astype("float32")
    y = np.zeros((num_v, feat_len), dtype="float32")

    if device == 'gpu':
        ir_dev = ir.Device(ir.GPU())
        ir_mtype = 'gpu/global'
        ir_shared_mtype = 'gpu/shared'
        ir_local_mtype = 'gpu/local'
    else:
        assert device == 'cpu'
        ir_dev = ir.Device(ir.CPU())
        ir_mtype = 'cpu'
        ir_shared_mtype = 'cpu'
        ir_local_mtype = 'cpu'

    ptr = ir.Array(ptr, ir_dev)
    idx = ir.Array(idx, ir_dev)
    x = ir.Array(x, ir_dev)
    w = ir.Array(w, ir_dev)
    w_attn_1 = ir.Array(w_attn_1, ir_dev)
    w_attn_2 = ir.Array(w_attn_2, ir_dev)
    y = ir.Array(y, ir_dev)

    test_num = 1000
    gat_layer(ptr, idx, x, w, w_attn_1, w_attn_2, y, num_v, num_e, feat_len,
              ir_dev, ir_mtype, ir_shared_mtype,
              ir_local_mtype)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        gat_layer(ptr, idx, x, w, w_attn_1, w_attn_2, y, num_v, num_e, feat_len,
                  ir_dev, ir_mtype, ir_shared_mtype, ir_local_mtype)
    t1 = time.time()

    print(f"Time = {(t1 - t0) / test_num * 1000} ms")
