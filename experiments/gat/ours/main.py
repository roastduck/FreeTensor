import sys
import time
import itertools
import numpy as np
import ir
from ir.libop import *
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


def compile_all(num_v, num_e, feat_len, device):
    mtype = device.main_mem_type()

    inf = float("inf")

    @ir.transform
    def inference(ptr, idx, feat, weight, attn_l, attn_r, y):
        ir.declare_var(ptr, (num_v + 1,), "int32", "input", mtype)
        ir.declare_var(idx, (num_e,), "int32", "input", mtype)
        ir.declare_var(feat, (num_v, feat_len), "float32", "input", mtype)
        ir.declare_var(weight, (feat_len, feat_len), "float32", "input", mtype)
        ir.declare_var(attn_l, (feat_len,), "float32", "input", mtype)
        ir.declare_var(attn_r, (feat_len,), "float32", "input", mtype)
        ir.declare_var(y, (num_v, feat_len), "float32", "output", mtype)

        feat2 = matmul(feat, weight)
        att_l = matmul(feat2, attn_l)
        att_r = matmul(feat2, attn_r)

        edge = ir.create_var((num_e,), "float32", mtype)
        edge_exp = ir.create_var((num_e,), "float32", mtype)
        'nid: Li'
        'no_deps: edge'
        'no_deps: edge_exp'
        'no_deps: idx'
        for i in range(num_v):
            edge_max = ir.create_var((), "float32", mtype)
            edge_max[()] = -inf
            'nid: Lk1'
            'no_deps: att_l'
            for k in range(ptr[i], ptr[i + 1]):
                e = ir.create_var((), "float32", mtype)
                e[()] = att_l[idx[k]] + att_r[i]
                edge[k] = ir.if_then_else(e[()] >= 0, e[()], e[()] * 0.1)
                edge_max[()] = ir.max(edge_max[()], edge[k])
            edge_sum = ir.create_var((), "float32", mtype)
            edge_sum[()] = 0
            'nid: Lk2'
            for k in range(ptr[i], ptr[i + 1]):
                edge_exp[k] = ir.exp(edge[k] - edge_max[()])
                edge_sum[()] += edge_exp[k]
            'nid: Lj'
            for j in range(feat_len):
                y[i, j] = 0
                'nid: Lk3'
                'no_deps: feat2'
                for k in range(ptr[i], ptr[i + 1]):
                    y[i, j] += feat2[idx[k], j] * edge_exp[k] / edge_sum[()]

    forward, backward, requires, privdes, _ = ir.grad(
        inference, set(["feat", "weight", "attn_l", "attn_r"]), set(["y"]))

    print("# Inference:")
    print(inference)
    s = ir.Schedule(inference)
    s.auto_schedule(device.target())
    f = ir.lower(s.func(), device.target())
    print(f)
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    inference_exe = ir.Driver(inference, code, device)

    return inference_exe, None, None
    #print("# Forward:")
    #print(forward)
    #s = ir.Schedule(forward)
    #s.auto_schedule(device.target())
    #f = ir.lower(s.func(), device.target())
    #print(f)
    #code = ir.codegen(f, device.target())
    #print(ir.debug.with_line_no(code))
    #forward_exe = ir.Driver(forward, code, device)

    #print("# Backward:")
    #print(backward)
    #s = ir.Schedule(backward)
    #s.auto_schedule(device.target())
    #print(s.ast())
    #f = ir.lower(s.func(), device.target())
    #print(f)
    #code = ir.codegen(f, device.target())
    #print(ir.debug.with_line_no(code))
    #backward_exe = ir.Driver(backward, code, device)

    #def run_backward(ptr, idx, x, w, w_attn_1, w_attn_2, y, d_y, d_x, d_w,
    #                 d_w_attn_1, d_w_attn_2):
    #    kvs = {}
    #    kvs[privdes['y']] = d_y
    #    kvs[requires['feat']] = d_x
    #    kvs[requires['weight']] = d_w
    #    kvs[requires['attn_l']] = d_w_attn_1
    #    kvs[requires['attn_r']] = d_w_attn_2
    #    backward_exe(ptr, idx, x, w, w_attn_1, w_attn_2, y, **kvs)

    #return inference_exe, forward_exe, run_backward


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu/gpu>")
        exit(-1)
    device = sys.argv[1]

    ptr = np.loadtxt("../ptr.in", dtype=np.int32)
    idx = np.loadtxt("../idx.in", dtype=np.int32)
    num_v = ptr.shape[0] - 1
    num_e = idx.shape[0]

    feat_len = 32
    ptr = ptr.astype("int32")
    idx = idx.astype("int32")
    x = np.loadtxt("../x.in").astype("float32")
    w = np.loadtxt("../w.in").astype("float32")
    w_attn_1 = np.loadtxt("../w_attn_1.in").astype("float32")
    w_attn_2 = np.loadtxt("../w_attn_2.in").astype("float32")
    y = np.zeros((num_v, feat_len), dtype="float32")
    d_x = np.zeros(x.shape, dtype='float32')
    d_w = np.zeros(w.shape, dtype='float32')
    d_w_attn_1 = np.zeros(w_attn_1.shape, dtype='float32')
    d_w_attn_2 = np.zeros(w_attn_2.shape, dtype='float32')
    d_y = np.loadtxt("../d_y.in").astype("float32")

    if device == 'gpu':
        ir_dev = ir.Device(ir.GPU())
    else:
        assert device == 'cpu'
        ir_dev = ir.Device(ir.CPU())

    ptr = ir.Array(ptr, ir_dev)
    idx = ir.Array(idx, ir_dev)
    x = ir.Array(x, ir_dev)
    w = ir.Array(w, ir_dev)
    w_attn_1 = ir.Array(w_attn_1, ir_dev)
    w_attn_2 = ir.Array(w_attn_2, ir_dev)
    y = ir.Array(y, ir_dev)
    d_x = ir.Array(d_x, ir_dev)
    d_w = ir.Array(d_w, ir_dev)
    d_w_attn_1 = ir.Array(d_w_attn_1, ir_dev)
    d_w_attn_2 = ir.Array(d_w_attn_2, ir_dev)
    d_y = ir.Array(d_y, ir_dev)

    inference, forward, backward = compile_all(num_v, num_e, feat_len, ir_dev)

    warmup_num = 10
    test_num = 1000

    for i in range(warmup_num):
        inference(ptr, idx, x, w, w_attn_1, w_attn_2, y)
        if i == 0:
            np.savetxt("y.out", y.numpy().reshape((num_v, feat_len)))
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        inference(ptr, idx, x, w, w_attn_1, w_attn_2, y)
    ir_dev.sync()
    t1 = time.time()

    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")

    #for i in range(warmup_num):
    #    forward(ptr, idx, x, w, w_attn_1, w_attn_2, y)
    #ir_dev.sync()
    #t0 = time.time()
    #for i in range(test_num):
    #    forward(ptr, idx, x, w, w_attn_1, w_attn_2, y)
    #ir_dev.sync()
    #t1 = time.time()

    #print(f"Forward Time = {(t1 - t0) / test_num * 1000} ms")

    #for i in range(warmup_num):
    #    backward(ptr, idx, x, w, w_attn_1, w_attn_2, y, d_y, d_x, d_w,
    #             d_w_attn_1, d_w_attn_2)
    #    if i == 0:
    #        np.savetxt("d_x.out", d_x.numpy().reshape((num_v, feat_len)))
    #        np.savetxt("d_w.out", d_w.numpy().reshape((feat_len, feat_len)))
    #        np.savetxt("d_w_attn_1.out", d_w_attn_1.numpy())
    #        np.savetxt("d_w_attn_2.out", d_w_attn_2.numpy())
    #ir_dev.sync()
    #t0 = time.time()
    #for i in range(test_num):
    #    backward(ptr, idx, x, w, w_attn_1, w_attn_2, y, d_y, d_x, d_w,
    #             d_w_attn_1, d_w_attn_2)
    #ir_dev.sync()
    #t1 = time.time()

    #print(f"Backward Time = {(t1 - t0) / test_num * 1000} ms")
