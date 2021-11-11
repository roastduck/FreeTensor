import sys
import time
import math
import numpy as np
import ir
import ir.debug


def compile_all(w, dilation, dilation_heads, n_heads, seq_len, feat_len,
                device):
    mtype = device.main_mem_type()

    sqrt_d = math.sqrt(feat_len)
    inf = float("inf")

    @ir.transform
    def inference(Q, K, V, Y):
        ir.declare_var(Q, (n_heads, seq_len, feat_len), "float32", "input",
                       mtype)
        ir.declare_var(K, (n_heads, seq_len, feat_len), "float32", "input",
                       mtype)
        ir.declare_var(V, (n_heads, seq_len, feat_len), "float32", "input",
                       mtype)
        ir.declare_var(Y, (n_heads, seq_len, feat_len), "float32", "output",
                       mtype)
        for i in range(n_heads):
            for j in range(seq_len):
                dot = ir.create_var((2 * w + 1,), "float32", mtype)
                for k in range(-w, w + 1):
                    dot[k + w] = 0
                    if j + ir.if_then_else(
                            i >= dilation_heads, k,
                            k * dilation) >= 0 and j + ir.if_then_else(
                                i >= dilation_heads, k, k * dilation) < seq_len:
                        for p in range(feat_len):
                            dot[k + w] += Q[i, j, p] * K[i, j + ir.if_then_else(
                                i >= dilation_heads, k, k * dilation), p]

                maxval = ir.create_var((), "float32", mtype)
                maxval[()] = -inf
                for k in range(2 * w + 1):
                    maxval[()] = ir.max(maxval[()], dot[k])
                expval = ir.create_var((2 * w + 1,), "float32", mtype)
                for k in range(2 * w + 1):
                    expval[k] = ir.exp(dot[k] - maxval[()])
                expsum = ir.create_var((), "float32", mtype)
                expsum[()] = 0
                for k in range(2 * w + 1):
                    expsum[()] += expval[k]
                attn = ir.create_var((2 * w + 1,), "float32", mtype)
                for k in range(2 * w + 1):
                    attn[k] = expval[k] / expsum[()] / sqrt_d

                for p in range(feat_len):
                    Y[i, j, p] = 0
                for k in range(-w, w + 1):
                    if j + ir.if_then_else(
                            i >= dilation_heads, k,
                            k * dilation) >= 0 and j + ir.if_then_else(
                                i >= dilation_heads, k, k * dilation) < seq_len:
                        for p in range(feat_len):
                            Y[i, j,
                              p] += attn[k + w] * V[i, j + ir.if_then_else(
                                  i >= dilation_heads, k, k * dilation), p]

    forward, backward, requires, privdes, _ = ir.grad(inference,
                                                      set(["Q", "K", "V"]),
                                                      set(["Y"]))

    print("# Inference:")
    print(inference)
    s = ir.Schedule(inference)
    s.auto_schedule(device.target())
    f = ir.lower(s.func(), device.target())
    print(f)
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    inference_exe = ir.Driver(inference, code, device)

    print("# Forward:")
    print(forward)
    s = ir.Schedule(forward)
    s.auto_schedule(device.target())
    f = ir.lower(s.func(), device.target())
    print(f)
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    forward_exe = ir.Driver(forward, code, device)

    print("# Backward:")
    print(backward)
    s = ir.Schedule(backward)
    s.auto_schedule(device.target())
    f = ir.lower(s.func(), device.target())
    print(f)
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    backward_exe = ir.Driver(backward, code, device)

    def run_backward(Q, K, V, Y, d_Y, d_Q, d_K, d_V):
        kvs = {}
        kvs[privdes['Y']] = d_Y
        kvs[requires['Q']] = d_Q
        kvs[requires['K']] = d_K
        kvs[requires['V']] = d_V
        backward_exe(Q, K, V, Y, **kvs)

    return inference_exe, forward_exe, run_backward


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu/gpu>")
        exit(-1)
    device = sys.argv[1]

    n_heads = 8
    seq_len = 10000
    feat_len = 512
    w = 32
    dilation = 4  # counts from 1
    dilation_heads = 2
    q = np.load("../q.in.npy").astype("float32")
    k = np.load("../k.in.npy").astype("float32")
    v = np.load("../v.in.npy").astype("float32")
    y = np.zeros((n_heads, seq_len, feat_len), dtype="float32")
    d_q = np.zeros(q.shape, dtype='float32')
    d_k = np.zeros(k.shape, dtype='float32')
    d_v = np.zeros(v.shape, dtype='float32')
    d_y = np.load("../d_y.in.npy").astype("float32")

    if device == 'gpu':
        ir_dev = ir.Device(ir.GPU())
    else:
        assert device == 'cpu'
        ir_dev = ir.Device(ir.CPU())

    q = ir.Array(q, ir_dev)
    k = ir.Array(k, ir_dev)
    v = ir.Array(v, ir_dev)
    y = ir.Array(y, ir_dev)
    d_q = ir.Array(d_q, ir_dev)
    d_k = ir.Array(d_k, ir_dev)
    d_v = ir.Array(d_v, ir_dev)
    d_y = ir.Array(d_y, ir_dev)

    inference, forward, backward = compile_all(w, dilation, dilation_heads,
                                               n_heads, seq_len, feat_len,
                                               ir_dev)

    warmup_num = 10
    test_num = 100

    for i in range(warmup_num):
        inference(q, k, v, y)
        if i == 0:
            np.save("y.out.npy",
                    y.numpy().reshape((n_heads, seq_len, feat_len)),
                    allow_pickle=False)
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        inference(q, k, v, y)
    ir_dev.sync()
    t1 = time.time()

    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        forward(q, k, v, y)
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        forward(q, k, v, y)
    ir_dev.sync()
    t1 = time.time()

    print(f"Forward Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        backward(q, k, v, y, d_y, d_q, d_k, d_v)
        if i == 0:
            np.save("d_q.out.npy",
                    d_q.numpy().reshape((n_heads, seq_len, feat_len)),
                    allow_pickle=False)
            np.save("d_k.out.npy",
                    d_k.numpy().reshape((n_heads, seq_len, feat_len)),
                    allow_pickle=False)
            np.save("d_v.out.npy",
                    d_v.numpy().reshape((n_heads, seq_len, feat_len)),
                    allow_pickle=False)
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        backward(q, k, v, y, d_y, d_q, d_k, d_v)
    ir_dev.sync()
    t1 = time.time()

    print(f"Backward Time = {(t1 - t0) / test_num * 1000} ms")
