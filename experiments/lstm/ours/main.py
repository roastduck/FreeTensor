import sys
import time
import itertools
import numpy as np
import freetensor as ft
from freetensor.libop import *
from freetensor import debug

sys.path.append('../..')
from common.numpy.io import load_txt, store_txt


def compile_all(n_faces, in_feats, out_feats, device, ad_save_all):
    mtype = device.main_mem_type()

    @ft.transform
    def inference(adj, x, w0, w1, w2, w3, y):
        adj: ft.Var[(n_faces, 3), "int32", "input", mtype]
        x: ft.Var[(n_faces, in_feats), "float32", "input", mtype]
        w0: ft.Var[(in_feats, out_feats), "float32", "input", mtype]
        w1: ft.Var[(in_feats, out_feats), "float32", "input", mtype]
        w2: ft.Var[(in_feats, out_feats), "float32", "input", mtype]
        w3: ft.Var[(in_feats, out_feats), "float32", "input", mtype]
        y: ft.Var[(n_faces, out_feats), "float32", "output", mtype]

        for i in range(n_faces):
            sum1 = zeros((in_feats,), "float32", mtype)
            sum2 = zeros((in_feats,), "float32", mtype)
            sum3 = zeros((in_feats,), "float32", mtype)
            for p in range(3):
                add_to(sum1, x[adj[i, p]])
                add_to(sum2, abs(sub(x[adj[i, p]], x[adj[i, (p + 1) % 3]])))
                add_to(sum3, abs(sub(x[adj[i, p]], x[i])))
            y0 = matmul(x[i], w0)
            y1 = matmul(sum1, w1)
            y2 = matmul(sum2, w2)
            y3 = matmul(sum3, w3)
            assign(y[i], add(add(add(y0, y1), y2), y3))

    print("# Inference:")
    print(inference)
    t0 = time.time()
    s = ft.Schedule(inference)
    s.auto_schedule(device.target())
    f = ft.lower(s.func(), device.target())
    code = ft.codegen(f, device.target())
    inference_exe = ft.Driver(inference, code, device)
    t1 = time.time()
    print(f)
    print(debug.with_line_no(code))
    print(f"Inference compiling time: {t1 - t0}s")

    forward, backward, requires, privdes, _ = ft.grad(
        inference, set(["x", "w0", "w1", "w2", "w3"]), set(["y"]),
        ft.GradTapeMode.All if ad_save_all else ft.GradTapeMode.NoReuseOnly)

    print("# Forward:")
    print(forward)
    s = ft.Schedule(forward)
    s.auto_schedule(device.target())
    f = ft.lower(s.func(), device.target())
    print(f)
    code = ft.codegen(f, device.target())
    print(debug.with_line_no(code))
    forward_exe = ft.Driver(forward, code, device)

    print("# Backward:")
    print(backward)
    s = ft.Schedule(backward)
    s.auto_schedule(device.target())
    f = ft.lower(s.func(), device.target())
    print(f)
    code = ft.codegen(f, device.target())
    print(debug.with_line_no(code))
    backward_exe = ft.Driver(backward, code, device)

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
    parser.add_argument('--profile-gpu',
                        action='store_true',
                        dest='profile_gpu')
    parser.add_argument('--ad-save-all',
                        action='store_true',
                        dest='ad_save_all')
    cmd_args = parser.parse_args()

    if cmd_args.profile_gpu:
        from common.gpu import profile_start, profile_stop

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
        ir_dev = ft.Device(ft.GPU())
    else:
        assert device == 'cpu'
        ir_dev = ft.Device(ft.CPU())

    adj = ft.Array(adj, ir_dev)
    x = ft.Array(x, ir_dev)
    w0 = ft.Array(w0, ir_dev)
    w1 = ft.Array(w1, ir_dev)
    w2 = ft.Array(w2, ir_dev)
    w3 = ft.Array(w3, ir_dev)
    y = ft.Array(y, ir_dev)
    d_x = ft.Array(d_x, ir_dev)
    d_w0 = ft.Array(d_w0, ir_dev)
    d_w1 = ft.Array(d_w1, ir_dev)
    d_w2 = ft.Array(d_w2, ir_dev)
    d_w3 = ft.Array(d_w3, ir_dev)
    d_y = ft.Array(d_y, ir_dev)

    inference, forward, backward = compile_all(n_faces, in_feats, out_feats,
                                               ir_dev, cmd_args.ad_save_all)

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
    if cmd_args.profile_gpu:
        profile_start()
    t0 = time.time()
    for i in range(test_num):
        inference(adj, x, w0, w1, w2, w3, y)
    ir_dev.sync()
    t1 = time.time()
    if cmd_args.profile_gpu:
        profile_stop()

    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")

    if cmd_args.profile_gpu:
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


def compile_all(in_feats, hidden_feats, length, device):
    mtype = device.main_mem_type()

    @ft.transform
    def inference(x, y, w, u, b):
        x: ft.Var[(length, in_feats), "float32", "input", mtype]
        y: ft.Var[(hidden_feats,), "float32", "output", mtype]
        w: ft.Var[(4, in_feats, hidden_feats), "float32", "input", mtype]
        u: ft.Var[(4, hidden_feats, hidden_feats), "float32", "input", mtype]
        b: ft.Var[(4, hidden_feats), "float32", "input", mtype]
        h = ft.empty((hidden_feats,), "float32", mtype)
        c = ft.empty((hidden_feats,), "float32", mtype)
        f = ft.empty((4, hidden_feats), "float32", mtype)

        for l in range(hidden_feats):
            c[l] = 0
            h[l] = 0
        "nid: K"
        for k in range(length):
            'nid: m_in'
            for m in range(4):
                'nid: l_in'
                for l in range(hidden_feats):
                    f[m][l] = b[m][l]
                    'nid: j_in'
                    for j in range(in_feats):
                        f[m][l] += w[m][j][l] * x[k][j]
                    'nid: j_hidden'
                    for j in range(hidden_feats):
                        f[m][l] += u[m][j][l] * h[j]
            "nid: ch"
            for l in range(hidden_feats):
                c[l] = ft.sigmoid(f[0][l]) * c[l] + ft.sigmoid(
                    f[1][l]) * ft.tanh(f[3][l])
                h[l] = ft.sigmoid(f[2][l]) * ft.tanh(c[l])
        assign(y, h)

    forward, backward, requires, provides, _ = ft.grad(inference,
                                                       {"x", "w", "u", "b"},
                                                       {"y"},
                                                       ft.GradTapeMode.All)

    print("# Inference:")
    print(inference)
    s = ft.Schedule(inference)
    s.auto_schedule(device.target())
    f = ft.lower(s.func(), device.target())
    print(f)
    code = ft.codegen(f, device.target())
    print(debug.with_line_no(code))
    inference_exe = ft.Driver(inference, code, device)

    print("# Forward:")
    print(forward)
    s = ft.Schedule(forward)
    s.auto_schedule(device.target())
    f = ft.lower(s.func(), device.target())
    print(f)
    code = ft.codegen(f, device.target())
    print(debug.with_line_no(code))
    forward_exe = ft.Driver(forward, code, device)

    print("# Backward:")
    print(backward)
    s = ft.Schedule(backward)
    s.auto_schedule(device.target())
    print(s.ast())
    f = ft.lower(s.func(), device.target())
    print(f)
    code = ft.codegen(f, device.target())
    print(debug.with_line_no(code))
    backward_exe = ft.Driver(backward, code, device)

    def run_backward(x, y, w, u, b, d_w, d_u, d_b, d_x, d_y):
        kvs = {}
        kvs[provides['y']] = d_y
        kvs[requires['x']] = d_x
        kvs[requires['w']] = d_w
        kvs[requires['u']] = d_u
        kvs[requires['b']] = d_b
        backward_exe(x, y, w, u, b, **kvs)

    return inference_exe, forward_exe, run_backward


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu/gpu>")
        exit(-1)
    device = sys.argv[1]

    x = np.loadtxt("../x.in").astype("float32")
    wf = np.loadtxt("../wf.in").astype("float32").transpose()
    wi = np.loadtxt("../wi.in").astype("float32").transpose()
    wo = np.loadtxt("../wo.in").astype("float32").transpose()
    wc = np.loadtxt("../wc.in").astype("float32").transpose()
    uf = np.loadtxt("../uf.in").astype("float32").transpose()
    ui = np.loadtxt("../ui.in").astype("float32").transpose()
    uo = np.loadtxt("../uo.in").astype("float32").transpose()
    uc = np.loadtxt("../uc.in").astype("float32").transpose()
    bf = np.loadtxt("../bf.in").astype("float32")
    bi = np.loadtxt("../bi.in").astype("float32")
    bo = np.loadtxt("../bo.in").astype("float32")
    bc = np.loadtxt("../bc.in").astype("float32")
    in_feats = x.shape[1]
    hidden_feats = uf.shape[0]
    length = x.shape[0]
    y = np.zeros((hidden_feats,), dtype="float32")
    w = np.zeros((4,) + wf.shape, dtype="float32")
    w[0] = wf
    w[1] = wi
    w[2] = wo
    w[3] = wc
    u = np.zeros((4,) + uf.shape, dtype="float32")
    u[0] = uf
    u[1] = ui
    u[2] = uo
    u[3] = uc
    b = np.zeros((4,) + bf.shape, dtype="float32")
    b[0] = bf
    b[1] = bi
    b[2] = bo
    b[3] = bc

    d_x = np.zeros(x.shape, dtype='float32')
    d_w = np.zeros(w.shape, dtype='float32')
    d_u = np.zeros(u.shape, dtype='float32')
    d_b = np.zeros(b.shape, dtype='float32')
    d_y = np.loadtxt("../d_y.in").astype("float32")

    if device == 'gpu':
        ir_dev = ft.Device(ft.GPU())
    else:
        assert device == 'cpu'
        ir_dev = ft.Device(ft.CPU())

    x = ft.Array(x, ir_dev)
    w = ft.Array(w, ir_dev)
    u = ft.Array(u, ir_dev)
    b = ft.Array(b, ir_dev)
    y = ft.Array(y, ir_dev)
    d_x = ft.Array(d_x, ir_dev)
    d_w = ft.Array(d_w, ir_dev)
    d_u = ft.Array(d_u, ir_dev)
    d_b = ft.Array(d_b, ir_dev)
    d_y = ft.Array(d_y, ir_dev)

    inference, forward, backward = compile_all(in_feats, hidden_feats, length,
                                               ir_dev)
    warmup_num = 10
    test_num = 10
    for i in range(warmup_num):
        inference(x, y, w, u, b)
        if i == 0:
            np.savetxt("y.out", y.numpy().reshape((hidden_feats,)))
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        inference(x, y, w, u, b)
    ir_dev.sync()
    t1 = time.time()

    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")
    #
    for i in range(warmup_num):
        forward(x, y, w, u, b)
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        forward(x, y, w, u, b)
    ir_dev.sync()
    t1 = time.time()

    print(f"Forward Time = {(t1 - t0) / test_num * 1000} ms")
    for i in range(warmup_num):
        backward(x, y, w, u, b, d_w, d_u, d_b, d_x, d_y)
        if i == 0:
            np.savetxt("d_x.out", d_x.numpy().reshape((length, in_feats)))
            np.savetxt(
                "d_wf.out",
                d_w.numpy().reshape((4, in_feats, hidden_feats))[0].transpose())
            np.savetxt(
                "d_wi.out",
                d_w.numpy().reshape((4, in_feats, hidden_feats))[1].transpose())
            np.savetxt(
                "d_wo.out",
                d_w.numpy().reshape((4, in_feats, hidden_feats))[2].transpose())
            np.savetxt(
                "d_wc.out",
                d_w.numpy().reshape((4, in_feats, hidden_feats))[3].transpose())
            np.savetxt(
                "d_uf.out",
                d_u.numpy().reshape(
                    (4, hidden_feats, hidden_feats))[0].transpose())
            np.savetxt(
                "d_ui.out",
                d_u.numpy().reshape(
                    (4, hidden_feats, hidden_feats))[1].transpose())
            np.savetxt(
                "d_uo.out",
                d_u.numpy().reshape(
                    (4, hidden_feats, hidden_feats))[2].transpose())
            np.savetxt(
                "d_uc.out",
                d_u.numpy().reshape(
                    (4, hidden_feats, hidden_feats))[3].transpose())
            np.savetxt("d_bf.out", d_b.numpy().reshape((4, hidden_feats))[0])
            np.savetxt("d_bi.out", d_b.numpy().reshape((4, hidden_feats))[1])
            np.savetxt("d_bo.out", d_b.numpy().reshape((4, hidden_feats))[2])
            np.savetxt("d_bc.out", d_b.numpy().reshape((4, hidden_feats))[3])
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        backward(x, y, w, u, b, d_w, d_u, d_b, d_x, d_y)
    ir_dev.sync()
    t1 = time.time()

    print(f"Backward Time = {(t1 - t0) / test_num * 1000} ms")
