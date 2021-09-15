import sys
import time
import math
import numpy as np
import ir
import ir.debug
from ir.libop import StaticType as T

jit_cache = {}


def transformer(q, k, v, y, w, dilation, dilation_heads, n_heads, seq_len,
                feat_len, device, mtype, local_mtype):
    if (w, n_heads, seq_len, feat_len) in jit_cache:
        exe = jit_cache[(w, n_heads, seq_len, feat_len)]

    else:

        sqrt_d = math.sqrt(feat_len)
        inf = float("inf")

        @ir.transform
        def f(Q, K, V, Y):
            ir.declare_var(Q, (n_heads, seq_len, feat_len), "float32", "input",
                           mtype)
            ir.declare_var(K, (n_heads, seq_len, feat_len), "float32", "input",
                           mtype)
            ir.declare_var(V, (n_heads, seq_len, feat_len), "float32", "input",
                           mtype)
            ir.declare_var(Y, (n_heads, seq_len, feat_len), "float32", "output",
                           mtype)
            "nid: Li"
            for i in range(n_heads):
                "nid: Lj"
                for j in range(seq_len):
                    dot = ir.create_var((2 * w + 1,), "float32", "cache",
                                        local_mtype)
                    "nid: Lk1"
                    for k in range(-w, w + 1):
                        dot[k + w] = 0
                        if i < dilation_heads:
                            if j + k >= 0 and j + k < seq_len:
                                "nid: Lp1"
                                for p in range(feat_len):
                                    dot[k + w] += Q[i, j, p] * K[i, j + k, p]
                        else:
                            if j + k * dilation >= 0 and j + k * dilation < seq_len:
                                "nid: Lp2"
                                for p in range(feat_len):
                                    dot[k +
                                        w] += Q[i, j,
                                                p] * K[i, j + k * dilation, p]

                    maxval = ir.create_var((), "float32", "cache", local_mtype)
                    maxval[()] = -inf
                    "nid: Lk2"
                    for k in range(2 * w + 1):
                        maxval[()] = ir.max(maxval, dot[k])
                    expval = ir.create_var((2 * w + 1,), "float32", "cache",
                                           local_mtype)
                    "nid: Lk3"
                    for k in range(2 * w + 1):
                        expval[k] = ir.exp(dot[k] - maxval[k])
                    expsum = ir.create_var((), "float32", "cache", local_mtype)
                    expsum[()] = 0
                    "nid: Lk4"
                    for k in range(2 * w + 1):
                        expsum[()] += expval[k]
                    attn = ir.create_var((2 * w + 1,), "float32", "cache",
                                         local_mtype)
                    "nid: Lk5"
                    for k in range(2 * w + 1):
                        attn[k] = expval[k] / expsum[()] / sqrt_d

                    "nid: Lp3"
                    for p in range(feat_len):
                        Y[i, j, p] = 0
                    if i < dilation_heads:
                        "nid: Lk6"
                        for k in range(-w, w + 1):
                            if j + k >= 0 and j + k < seq_len:
                                "nid: Lp4"
                                for p in range(feat_len):
                                    Y[i, j, p] += attn[k + w] * V[i, j + k, p]
                    else:
                        "nid: Lk7"
                        for k in range(-w, w + 1):
                            if j + k * dilation >= 0 and j + k * dilation < seq_len:
                                "nid: Lp5"
                                for p in range(feat_len):
                                    Y[i, j,
                                      p] += attn[k + w] * V[i, j + k * dilation,
                                                            p]

        s = ir.Schedule(f)
        print(s.ast())
        if device.target().type() == ir.TargetType.CPU:
            Lij = s.merge('Li', 'Lj')
            s.parallelize(Lij, 'openmp')
        else:
            p0, p1 = s.split('Lp1', 32)
            s.reorder([p1, p0])
            init, final, _ = s.cache_reduction(p0, "$:dot", "gpu/shared")
            final = s.move_to(final, ir.MoveToSide.After, p1)
            p0, p1 = s.split('Lp2', 32)
            s.reorder([p1, p0])
            init, final, _ = s.cache_reduction(p0, "$:dot", "gpu/shared")
            final = s.move_to(final, ir.MoveToSide.After, p1)
            p0, p1 = s.split('Lp3', 32)
            p0, p1 = s.split('Lp4', 32)
            p0, p1 = s.split('Lp5', 32)
            s.cache(
                s.find(lambda x: x.nid() == 'Lj').node().body, 'Y',
                'gpu/shared')
            s.unroll('Lk1')
            s.unroll('Lk2')
            s.unroll('Lk3')
            s.unroll('Lk4')
            s.unroll('Lk5')
            s.unroll('Lk6')
            s.unroll('Lk7')
            Lij = s.merge('Li', 'Lj')
            blk, thr = s.split(Lij, 8)
            s.parallelize(blk, 'blockIdx.x')
            s.parallelize(thr, 'threadIdx.y')
            s.parallelize('Lp1.1', 'threadIdx.x')
            s.parallelize('Lp2.1', 'threadIdx.x')
            s.parallelize('Lp3.1', 'threadIdx.x')
            s.parallelize('Lp4.1', 'threadIdx.x')
            s.parallelize('Lp5.1', 'threadIdx.x')
        f = ir.lower(s.func(), device.target())
        print(f)
        code = ir.codegen(f, device.target())
        print(ir.debug.with_line_no(code))
        exe = ir.Driver(f, code, device)
        exe.set_params(q, k, v, y)
        # TODO: do not set_params here
        jit_cache[(w, n_heads, seq_len, feat_len)] = exe

    exe.run()
    exe.sync()


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
    q = np.random.uniform(size=(n_heads, seq_len, feat_len)).astype("float32")
    k = np.random.uniform(size=(n_heads, seq_len, feat_len)).astype("float32")
    v = np.random.uniform(size=(n_heads, seq_len, feat_len)).astype("float32")
    y = np.zeros((n_heads, seq_len, feat_len), dtype="float32")

    if device == 'gpu':
        ir_dev = ir.Device(ir.GPU())
        ir_mtype = 'gpu/global'
        ir_local_mtype = 'gpu/local'
    else:
        assert device == 'cpu'
        ir_dev = ir.Device(ir.CPU())
        ir_mtype = 'cpu'
        ir_local_mtype = 'cpu'

    q = ir.Array(q, ir_dev)
    k = ir.Array(k, ir_dev)
    v = ir.Array(v, ir_dev)
    y = ir.Array(y, ir_dev)

    test_num = 100
    transformer(q, k, v, y, w, dilation, dilation_heads, n_heads, seq_len,
                feat_len, ir_dev, ir_mtype, ir_local_mtype)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        transformer(q, k, v, y, w, dilation, dilation_heads, n_heads, seq_len,
                    feat_len, ir_dev, ir_mtype, ir_local_mtype)
    t1 = time.time()

    print(f"Time = {(t1 - t0) / test_num * 1000} ms")
