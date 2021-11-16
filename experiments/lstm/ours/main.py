import sys
import time
import itertools
import numpy as np
import ir
from ir.libop import *
import ir.debug

# len = 256
# @ir.transform
# def foo(y, h, c):
#     ir.declare_var(y, (len, ), "float32", "input", "cpu")
#     ir.declare_var(h, (len, ), "float32", "input", "cpu")
#     ir.declare_var(c, (len, ), "float32", "output", "cpu")
#     f = zeros((len, ), "float32", "cpu")()
#     u = zeros((len, ), "float32", "cpu")()
#     for l in range(len):
#         for j in range(len):
#             f[j] = 0
#             for k in range(len):
#                 f[j] += u[j]
#                 f[j] += 1
#             u[j] = f[j]
#     assign(y, f)
#     return y
#
# a = np.zeros((len,))
# b = np.zeros((len,))
# c = np.zeros((len,))
# d = np.zeros((len,))
# e = np.zeros((len,))
# f = np.zeros((len,))
# device = ir_dev = ir.Device(ir.CPU())
# a = ir.Array(a, ir_dev)
# b = ir.Array(b, ir_dev)
# c = ir.Array(c, ir_dev)
# d = ir.Array(d, ir_dev)
# e = ir.Array(e, ir_dev)
# f = ir.Array(f, ir_dev)
#
# s = ir.Schedule(foo)
# s.auto_schedule(ir.CPU())
# print(s.ast())
# f = ir.lower(s.func(), device.target())
# print(f)
# code = ir.codegen(f, device.target())
# print(ir.debug.with_line_no(code))

def compile_all(in_feats, hidden_feats, length, device):
    mtype = device.main_mem_type()

    @ir.transform
    def inference(x, y, w, u, b):
        ir.declare_var(x, (length, in_feats), "float32", "input", mtype)
        ir.declare_var(y, (hidden_feats, ), "float32", "output", mtype)
        ir.declare_var(w, (4, in_feats, hidden_feats), "float32", "input", mtype)
        ir.declare_var(u, (4, hidden_feats, hidden_feats), "float32", "input", mtype)
        ir.declare_var(b, (4, hidden_feats, ), "float32", "input", mtype)
        h = zeros((hidden_feats, ), "float32", mtype)()
        c = zeros((hidden_feats, ), "float32", mtype)()
        f = zeros((4, hidden_feats, ), "float32", mtype)()
        "nid: K"
        for k in range(length):
            'nid: m_in'
            for m in range(4):
                'nid: l_in'
                for l in range(hidden_feats):
                    f[m][l] = 0
                    'nid: j_in'
                    for j in range(in_feats):
                        f[m][l] += w[m][j][l] * x[k][j]
            "nid: m_hidden"
            for m in range(4):
                'nid: l_hidden'
                for l in range(hidden_feats):
                    "nid: j_hidden"
                    for j in range(hidden_feats):
                        f[m][l] += u[m][j][l] * h[j]
            "nid: m_b"
            for m in range(4):
                'nid: l_b'
                for l in range(hidden_feats):
                    f[m][l] += b[m][l]
            "nid: ch"
            for l in range(hidden_feats):
                c[l] = ir.sigmoid(f[0][l]) * c[l] + ir.sigmoid(f[1][l]) * ir.tanh(f[3][l])
                h[l] = ir.sigmoid(f[2][l]) * ir.tanh(c[l])
        assign(y, h)

    forward, backward, requires, provides, _ = ir.grad(
        inference, {"x", "w", "u", "b"}, {"y"})

    print("# Inference:")
    print(inference)
    s = ir.Schedule(inference)
    '''
    s.auto_use_lib(device.target())
    s.auto_fuse(device.target())
    s.split("fused.fused.l_in.l_hidden.l_b", 8)
    s.parallelize("fused.fused.l_in.l_hidden.l_b.0", "blockIdx.x")
    s.parallelize("fused.fused.l_in.l_hidden.l_b.1", "threadIdx.x")
    s.parallelize("fused.fused.m_in.m_hidden.m_b", "blockIdx.y")
    s.parallelize("fused.#2:recur:L_elem.#6:recur:L_elem", "threadIdx.x")
    s.parallelize("ch", "threadIdx.x")
    s.parallelize("#15:L_elem", "threadIdx.x")
    s.auto_set_mem_type(device.target())
    s.auto_unroll(device.target())
    s.set_mem_type("#6:y", "gpu/global")
    '''
    f = ir.lower(s.func(), device.target())
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    inference_exe = ir.Driver(inference, code, device)

    print("# Forward:")
    print(forward)
    s = ir.Schedule(forward)
    '''
    s.auto_use_lib(device.target())
    s.auto_fuse(device.target())
    s.split("fused.fused.l_in.l_hidden.l_b", 8)
    s.parallelize("fused.fused.l_in.l_hidden.l_b.0", "blockIdx.x")
    s.parallelize("fused.fused.l_in.l_hidden.l_b.1", "threadIdx.x")
    s.parallelize("fused.fused.m_in.m_hidden.m_b", "blockIdx.y")
    s.parallelize("fused.#2:recur:L_elem.#6:recur:L_elem", "threadIdx.x")
    s.parallelize("ch", "threadIdx.x")
    s.parallelize("#15:L_elem", "threadIdx.x")
    s.auto_set_mem_type(device.target())
    s.auto_unroll(device.target())
    s.set_mem_type("#6:y", "gpu/global")
    '''
    f = ir.lower(s.func(), device.target())
    print(f)
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    forward_exe = ir.Driver(forward, code, device)

    print("# Backward:")
    print(backward)
    s = ir.Schedule(backward)
    print(s.ast())
    '''
    s.auto_use_lib(device.target())
    s.auto_fuse(device.target())
    print(s.ast())
    s.parallelize("#133", "threadIdx.y")
    s.parallelize("#132", "threadIdx.x")
    s.parallelize("#128", "blockIdx.x")
    s.parallelize("#127", "threadIdx.x")
    s.parallelize("fused.#123.#118", "threadIdx.x")
    s.parallelize("#112", "threadIdx.x")
    s.parallelize("#109", "blockIdx.x")
    s.parallelize("#108", "threadIdx.x")
    s.parallelize("fused.#28.#30", "threadIdx.x")
    s.split("fused.fused.#38.#42.#45", 16)
    s.parallelize("fused.fused.#38.#42.#45.0", "blockIdx.x")
    s.parallelize("fused.fused.#38.#42.#45.1", "threadIdx.x")
    s.parallelize("fused.fused.#39.#43.#46", "blockIdx.y")
    s.parallelize("#50", "threadIdx.x")
    s.parallelize("#15:L_elem", "threadIdx.x")
    s.parallelize("ch", "threadIdx.x")
    s.split("fused.fused.l_b.l_hidden.l_in", 16)
    s.parallelize("fused.fused.l_b.l_hidden.l_in.0", "blockIdx.x")
    s.parallelize("fused.fused.l_b.l_hidden.l_in.1", "threadIdx.x")
    s.parallelize("fused.fused.m_b.m_hidden.m_in", "blockIdx.y")
    s.parallelize("#10:recur:recur:L_elem", "threadIdx.x")
    s.parallelize("#10:recur:L_elem", "threadIdx.y")
    s.parallelize("fused.#6:recur:L_elem.#2:recur:L_elem", "threadIdx.x")
    s.auto_set_mem_type(device.target())
    s.auto_unroll(device.target())
    '''
    f = ir.lower(s.func(), device.target())
    print(f)
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    backward_exe = ir.Driver(backward, code, device)
    #
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
    y = np.zeros((hidden_feats, ), dtype="float32")
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

    print(w[0])
    print("Test begin")
    if device == 'gpu':
        ir_dev = ir.Device(ir.GPU())
    else:
        assert device == 'cpu'
        ir_dev = ir.Device(ir.CPU())

    x = ir.Array(x, ir_dev)
    w = ir.Array(w, ir_dev)
    u = ir.Array(u, ir_dev)
    b = ir.Array(b, ir_dev)
    y = ir.Array(y, ir_dev)
    d_x = ir.Array(d_x, ir_dev)
    d_w = ir.Array(d_w, ir_dev)
    d_u = ir.Array(d_u, ir_dev)
    d_b = ir.Array(d_b, ir_dev)
    d_y = ir.Array(d_y, ir_dev)

    inference, forward, backward = compile_all(in_feats, hidden_feats, length, ir_dev)
    warmup_num = 10
    test_num = 10
    for i in range(warmup_num):
        inference(x, y, w, u, b)
        if i == 0:
            # print(x.numpy().tolist())
            # print(w.numpy().reshape(4, in_feats, hidden_feats).tolist())
            # print(u.numpy().reshape(4, hidden_feats, hidden_feats).tolist())
            # print(b.numpy().reshape(4, hidden_feats).tolist())
            # print(y.numpy().tolist())
            np.savetxt("y.out", y.numpy().reshape((hidden_feats, )))
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
            np.savetxt("d_wf.out", d_w.numpy().reshape((4, in_feats, hidden_feats))[0].transpose())
            np.savetxt("d_wi.out", d_w.numpy().reshape((4, in_feats, hidden_feats))[1].transpose())
            np.savetxt("d_wo.out", d_w.numpy().reshape((4, in_feats, hidden_feats))[2].transpose())
            np.savetxt("d_wc.out", d_w.numpy().reshape((4, in_feats, hidden_feats))[3].transpose())
            np.savetxt("d_uf.out", d_u.numpy().reshape((4, hidden_feats, hidden_feats))[0].transpose())
            np.savetxt("d_ui.out", d_u.numpy().reshape((4, hidden_feats, hidden_feats))[1].transpose())
            np.savetxt("d_uo.out", d_u.numpy().reshape((4, hidden_feats, hidden_feats))[2].transpose())
            np.savetxt("d_uc.out", d_u.numpy().reshape((4, hidden_feats, hidden_feats))[3].transpose())
            np.savetxt("d_bf.out", d_b.numpy().reshape((4, hidden_feats, ))[0])
            np.savetxt("d_bi.out", d_b.numpy().reshape((4, hidden_feats, ))[1])
            np.savetxt("d_bo.out", d_b.numpy().reshape((4, hidden_feats, ))[2])
            np.savetxt("d_bc.out", d_b.numpy().reshape((4, hidden_feats, ))[3])
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        backward(x, y, w, u, b, d_w, d_u, d_b, d_x, d_y)
    ir_dev.sync()
    t1 = time.time()

    print(f"Backward Time = {(t1 - t0) / test_num * 1000} ms")
