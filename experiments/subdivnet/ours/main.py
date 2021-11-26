import sys
import time
import itertools
import numpy as np
import ir
from ir.libop import *
import ir.debug


def compile_all(n_faces, in_feats, out_feats, device):
    mtype = device.main_mem_type()

    @ir.transform
    def inference(adj, x, w0, w1, w2, w3, y):
        ir.declare_var(adj, (n_faces, 3), "int32", "input", mtype)
        ir.declare_var(x, (n_faces, in_feats), "float32", "input", mtype)
        ir.declare_var(w0, (in_feats, out_feats), "float32", "input", mtype)
        ir.declare_var(w1, (in_feats, out_feats), "float32", "input", mtype)
        ir.declare_var(w2, (in_feats, out_feats), "float32", "input", mtype)
        ir.declare_var(w3, (in_feats, out_feats), "float32", "input", mtype)
        ir.declare_var(y, (n_faces, out_feats), "float32", "output", mtype)

        for i in range(n_faces):
            sum1 = zeros((in_feats,), "float32", mtype)()
            sum2 = zeros((in_feats,), "float32", mtype)()
            sum3 = zeros((in_feats,), "float32", mtype)()
            for p in range(3):
                add_to(sum1, x[adj[i, p]])
                add_to(sum2, abs(sub(x[adj[i, p]], x[adj[i, (p + 1) % 3]])))
                add_to(sum3, abs(sub(x[adj[i, p]], x[i])))
            y0 = matmul(x[i], w0)
            y1 = matmul(sum1, w1)
            y2 = matmul(sum2, w2)
            y3 = matmul(sum3, w3)
            assign(y[i], add(add(add(y0, y1), y2), y3))

    forward, backward, requires, privdes, _ = ir.grad(
        inference, set(["x", "w0", "w1", "w2", "w3"]), set(["y"]))

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

    def run_backward(adj, x, w0, w1, w2, w3, y, d_y, d_x, d_w0, d_w1, d_w2,
                     d_w3):
        kvs = {}
        kvs[privdes['y']] = d_y
        kvs[requires['x']] = d_x
        kvs[requires['w0']] = d_w0
        kvs[requires['w1']] = d_w1
        kvs[requires['w2']] = d_w2
        kvs[requires['w3']] = d_w3
        backward_exe(adj, x, w0, w1, w2, w3, y, **kvs)

    return inference_exe, forward_exe, run_backward


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu/gpu>")
        exit(-1)
    device = sys.argv[1]

    adj = np.loadtxt("../adj.in", dtype=np.int32)
    n_faces = adj.shape[0]
    in_feats = 13
    out_feats = 64
    x = np.loadtxt("../x.in").astype("float32")
    w0 = np.loadtxt("../w0.in").astype("float32")
    w1 = np.loadtxt("../w1.in").astype("float32")
    w2 = np.loadtxt("../w2.in").astype("float32")
    w3 = np.loadtxt("../w3.in").astype("float32")
    y = np.zeros((n_faces, out_feats), dtype="float32")
    d_x = np.zeros(x.shape, dtype='float32')
    d_w0 = np.zeros(w0.shape, dtype='float32')
    d_w1 = np.zeros(w1.shape, dtype='float32')
    d_w2 = np.zeros(w2.shape, dtype='float32')
    d_w3 = np.zeros(w3.shape, dtype='float32')
    d_y = np.loadtxt("../d_y.in").astype("float32")

    if device == 'gpu':
        ir_dev = ir.Device(ir.GPU())
    else:
        assert device == 'cpu'
        ir_dev = ir.Device(ir.CPU())

    adj = ir.Array(adj, ir_dev)
    x = ir.Array(x, ir_dev)
    w0 = ir.Array(w0, ir_dev)
    w1 = ir.Array(w1, ir_dev)
    w2 = ir.Array(w2, ir_dev)
    w3 = ir.Array(w3, ir_dev)
    y = ir.Array(y, ir_dev)
    d_x = ir.Array(d_x, ir_dev)
    d_w0 = ir.Array(d_w0, ir_dev)
    d_w1 = ir.Array(d_w1, ir_dev)
    d_w2 = ir.Array(d_w2, ir_dev)
    d_w3 = ir.Array(d_w3, ir_dev)
    d_y = ir.Array(d_y, ir_dev)

    inference, forward, backward = compile_all(n_faces, in_feats, out_feats,
                                               ir_dev)

    warmup_num = 10
    test_num = 1000

    for i in range(warmup_num):
        inference(adj, x, w0, w1, w2, w3, y)
        if i == 0:
            np.savetxt("y.out", y.numpy().reshape((n_faces, out_feats)))
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        inference(adj, x, w0, w1, w2, w3, y)
    ir_dev.sync()
    t1 = time.time()

    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        forward(adj, x, w0, w1, w2, w3, y)
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        forward(adj, x, w0, w1, w2, w3, y)
    ir_dev.sync()
    t1 = time.time()

    print(f"Forward Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        backward(adj, x, w0, w1, w2, w3, y, d_y, d_x, d_w0, d_w1, d_w2, d_w3)
        if i == 0:
            np.savetxt("d_x.out", d_x.numpy().reshape((n_faces, in_feats)))
            np.savetxt("d_w0.out", d_w0.numpy().reshape((in_feats, out_feats)))
            np.savetxt("d_w1.out", d_w1.numpy().reshape((in_feats, out_feats)))
            np.savetxt("d_w2.out", d_w2.numpy().reshape((in_feats, out_feats)))
            np.savetxt("d_w3.out", d_w3.numpy().reshape((in_feats, out_feats)))
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        backward(adj, x, w0, w1, w2, w3, y, d_y, d_x, d_w0, d_w1, d_w2, d_w3)
    ir_dev.sync()
    t1 = time.time()

    print(f"Backward Time = {(t1 - t0) / test_num * 1000} ms")
