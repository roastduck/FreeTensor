import sys
import time
import itertools
import argparse
import numpy as np
import ir
from ir.libop import *
import ir.debug

sys.path.append('../..')
from common.numpy.io import load_txt, store_txt


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
    t0 = time.time()
    s = ir.Schedule(inference)
    s.auto_schedule(device.target())
    f = ir.lower(s.func(), device.target())
    code = ir.codegen(f, device.target())
    inference_exe = ir.Driver(inference, code, device)
    t1 = time.time()
    print(f)
    print(ir.debug.with_line_no(code))
    print(f"Inference compiling time: {t1 - t0}s")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('target', nargs='?')
    parser.add_argument('--warmup-repeat',
                        type=int,
                        default=10,
                        dest='warmup_num')
    parser.add_argument('--timing-repeat',
                        type=int,
                        default=100,
                        dest='test_num')
    parser.add_argument('--infer-only', action='store_true', dest='infer_only')
    cmd_args = parser.parse_args()

    device = cmd_args.target

    adj = load_txt("../adj.in", "int32")
    n_faces = adj.shape[0]
    in_feats = 13
    out_feats = 64
    x = load_txt("../x.in", "float32")
    w0 = load_txt("../w0.in", "float32")
    w1 = load_txt("../w1.in", "float32")
    w2 = load_txt("../w2.in", "float32")
    w3 = load_txt("../w3.in", "float32")
    y = np.zeros((n_faces, out_feats), dtype="float32")
    d_x = np.zeros(x.shape, dtype='float32')
    d_w0 = np.zeros(w0.shape, dtype='float32')
    d_w1 = np.zeros(w1.shape, dtype='float32')
    d_w2 = np.zeros(w2.shape, dtype='float32')
    d_w3 = np.zeros(w3.shape, dtype='float32')
    d_y = load_txt("../d_y.in", "float32")

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

    print(
        f"{cmd_args.warmup_num} warmup, {cmd_args.test_num} repeats for evalution"
    )
    warmup_num = cmd_args.warmup_num
    test_num = cmd_args.test_num

    for i in range(warmup_num):
        inference(adj, x, w0, w1, w2, w3, y)
        if i == 0:
            store_txt("y.out", y.numpy().reshape((n_faces, out_feats)))
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        inference(adj, x, w0, w1, w2, w3, y)
    ir_dev.sync()
    t1 = time.time()

    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")

    if cmd_args.infer_only:
        exit(0)

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
            store_txt("d_x.out", d_x.numpy().reshape((n_faces, in_feats)))
            store_txt("d_w0.out", d_w0.numpy().reshape((in_feats, out_feats)))
            store_txt("d_w1.out", d_w1.numpy().reshape((in_feats, out_feats)))
            store_txt("d_w2.out", d_w2.numpy().reshape((in_feats, out_feats)))
            store_txt("d_w3.out", d_w3.numpy().reshape((in_feats, out_feats)))
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        backward(adj, x, w0, w1, w2, w3, y, d_y, d_x, d_w0, d_w1, d_w2, d_w3)
    ir_dev.sync()
    t1 = time.time()

    print(f"Backward Time = {(t1 - t0) / test_num * 1000} ms")
