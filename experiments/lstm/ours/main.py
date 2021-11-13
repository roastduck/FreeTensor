import sys
import time
import itertools
import numpy as np
import ir
from ir.libop import *
import ir.debug


def compile_all(in_feats, hidden_feats, length, device):
    mtype = device.main_mem_type()

    @ir.transform
    def inference(x, y, wi, ui, bi, wf, uf, bf, wc, uc, bc, wo, uo, bo):
        ir.declare_var(x, (length, in_feats), "float32", "input", mtype)
        ir.declare_var(y, (hidden_feats, ), "float32", "output", mtype)
        ir.declare_var(wf, (hidden_feats, in_feats), "float32", "input", mtype)
        ir.declare_var(wi, (hidden_feats, in_feats), "float32", "input", mtype)
        ir.declare_var(wo, (hidden_feats, in_feats), "float32", "input", mtype)
        ir.declare_var(wc, (hidden_feats, in_feats), "float32", "input", mtype)
        ir.declare_var(uf, (hidden_feats, hidden_feats), "float32", "input", mtype)
        ir.declare_var(ui, (hidden_feats, hidden_feats), "float32", "input", mtype)
        ir.declare_var(uo, (hidden_feats, hidden_feats), "float32", "input", mtype)
        ir.declare_var(uc, (hidden_feats, hidden_feats), "float32", "input", mtype)
        ir.declare_var(bf, (hidden_feats, ), "float32", "input", mtype)
        ir.declare_var(bi, (hidden_feats, ), "float32", "input", mtype)
        ir.declare_var(bo, (hidden_feats, ), "float32", "input", mtype)
        ir.declare_var(bc, (hidden_feats, ), "float32", "input", mtype)
        h = zeros((hidden_feats, ), "float32", mtype)()
        c = zeros((hidden_feats, ), "float32", mtype)()

        for k in range(length):
            f = sigmoid(add(add(matmul(wf, x[k]), matmul(uf, h)), bf))
            i = sigmoid(add(add(matmul(wi, x[k]), matmul(ui, h)), bi))
            o = sigmoid(add(add(matmul(wo, x[k]), matmul(uo, h)), bo))
            cc = tanh(add(add(matmul(wc, x[k]), matmul(uc, h)), bc))
            assign(c, add(mul(f, c), mul(i, cc)))
            assign(h, mul(o, tanh(c)))

        assign(y, h)

    forward, backward, requires, provides, _ = ir.grad(
        inference, {"x", "wf", "wi", "wo", "wc", "uf", "ui", "uo", "uc", "bf", "bi", "bo", "bc"}, {"y"})

    print("# Inference:")
    print(inference)
    s = ir.Schedule(inference)
    s.auto_schedule(device.target())
    s.set_mem_type("#2:y", "gpu/global")
    s.set_mem_type("#6:y", "gpu/global")
    s.set_mem_type("#11:a:einsum:Y", "gpu/global")
    s.set_mem_type("#11:b:einsum:Y", "gpu/global")
    s.set_mem_type("#37:a:einsum:Y", "gpu/global")
    s.set_mem_type("#37:b:einsum:Y", "gpu/global")
    s.set_mem_type("#63:a:einsum:Y", "gpu/global")
    s.set_mem_type("#63:b:einsum:Y", "gpu/global")
    s.set_mem_type("#126:b:y", "gpu/global")
    s.set_mem_type("#129:x:out", "gpu/global")
    s.set_mem_type("#89:a:einsum:Y", "gpu/global")
    s.set_mem_type("#89:b:einsum:Y", "gpu/global")
    s.set_mem_type("#107:x:out", "gpu/global")
    s.set_mem_type("#115:a:out", "gpu/global")
    f = ir.lower(s.func(), device.target())
    print(f)
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    inference_exe = ir.Driver(inference, code, device)

    print("# Forward:")
    print(forward)
    s = ir.Schedule(forward)
    s.auto_schedule(device.target())
    s.set_mem_type("#2:y", "gpu/global")
    s.set_mem_type("#6:y", "gpu/global")
    s.set_mem_type("#11:a:einsum:Y", "gpu/global")
    s.set_mem_type("#11:b:einsum:Y", "gpu/global")
    s.set_mem_type("#37:a:einsum:Y", "gpu/global")
    s.set_mem_type("#37:b:einsum:Y", "gpu/global")
    s.set_mem_type("#63:a:einsum:Y", "gpu/global")
    s.set_mem_type("#63:b:einsum:Y", "gpu/global")
    s.set_mem_type("#126:b:y", "gpu/global")
    s.set_mem_type("#129:x:out", "gpu/global")
    s.set_mem_type("#89:a:einsum:Y", "gpu/global")
    s.set_mem_type("#89:b:einsum:Y", "gpu/global")
    s.set_mem_type("#107:x:out", "gpu/global")
    s.set_mem_type("#115:a:out", "gpu/global")
    f = ir.lower(s.func(), device.target())
    print(f)
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    forward_exe = ir.Driver(forward, code, device)
    #
    # print("# Backward:")
    # print(backward)
    # s = ir.Schedule(backward)
    # s.auto_set_mem_type(device.target())
    # s.auto_schedule(device.target())
    # f = ir.lower(s.func(), device.target())
    # print(f)
    # code = ir.codegen(f, device.target())
    # print(ir.debug.with_line_no(code))
    # backward_exe = ir.Driver(backward, code, device)
    # #
    # def run_backward(x, y, wi, ui, bi, wf, uf, bf, wc, uc, bc, wo, uo, bo,
    #                  d_wi, d_ui, d_bi, d_wf, d_uf, d_bf, d_wc, d_uc, d_bc, d_wo, d_uo, d_bo, d_x, d_y):
    #     kvs = {}
    #     kvs[provides['y']] = d_y
    #     kvs[requires['x']] = d_x
    #     kvs[requires['wf']] = d_wf
    #     kvs[requires['wi']] = d_wi
    #     kvs[requires['wo']] = d_wo
    #     kvs[requires['wc']] = d_wc
    #     kvs[requires['uf']] = d_uf
    #     kvs[requires['ui']] = d_ui
    #     kvs[requires['uo']] = d_uo
    #     kvs[requires['uc']] = d_uc
    #     kvs[requires['bf']] = d_bf
    #     kvs[requires['bi']] = d_bi
    #     kvs[requires['bo']] = d_bo
    #     kvs[requires['bc']] = d_bc
    #     backward_exe(x, y, wi, ui, bi, wf, uf, bf, wc, uc, bc, wo, uo, bo, **kvs)

    return inference_exe, forward_exe#, run_backward

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu/gpu>")
        exit(-1)
    device = sys.argv[1]

    in_feats = 4
    hidden_feats = 256
    length = 100
    x = np.loadtxt("../x.in").astype("float32")
    wf = np.loadtxt("../wf.in").astype("float32")
    wi = np.loadtxt("../wi.in").astype("float32")
    wo = np.loadtxt("../wo.in").astype("float32")
    wc = np.loadtxt("../wc.in").astype("float32")
    uf = np.loadtxt("../uf.in").astype("float32")
    ui = np.loadtxt("../ui.in").astype("float32")
    uo = np.loadtxt("../uo.in").astype("float32")
    uc = np.loadtxt("../uc.in").astype("float32")
    bf = np.loadtxt("../bf.in").astype("float32")
    bi = np.loadtxt("../bi.in").astype("float32")
    bo = np.loadtxt("../bo.in").astype("float32")
    bc = np.loadtxt("../bc.in").astype("float32")
    y = np.zeros((hidden_feats, ), dtype="float32")

    d_x = np.zeros(x.shape, dtype='float32')
    d_wf = np.zeros(wf.shape, dtype='float32')
    d_wi = np.zeros(wi.shape, dtype='float32')
    d_wo = np.zeros(wo.shape, dtype='float32')
    d_wc = np.zeros(wc.shape, dtype='float32')
    d_uf = np.zeros(uf.shape, dtype='float32')
    d_ui = np.zeros(ui.shape, dtype='float32')
    d_uo = np.zeros(uo.shape, dtype='float32')
    d_uc = np.zeros(uc.shape, dtype='float32')
    d_bf = np.zeros(bf.shape, dtype='float32')
    d_bi = np.zeros(bi.shape, dtype='float32')
    d_bo = np.zeros(bo.shape, dtype='float32')
    d_bc = np.zeros(bc.shape, dtype='float32')
    d_y = np.loadtxt("../d_y.in").astype("float32")

    if device == 'gpu':
        ir_dev = ir.Device(ir.GPU())
    else:
        assert device == 'cpu'
        ir_dev = ir.Device(ir.CPU())

    x = ir.Array(x, ir_dev)
    wf = ir.Array(wf, ir_dev)
    wi = ir.Array(wi, ir_dev)
    wo = ir.Array(wo, ir_dev)
    wc = ir.Array(wc, ir_dev)
    uf = ir.Array(uf, ir_dev)
    ui = ir.Array(ui, ir_dev)
    uo = ir.Array(uo, ir_dev)
    uc = ir.Array(uc, ir_dev)
    bf = ir.Array(bf, ir_dev)
    bi = ir.Array(bi, ir_dev)
    bo = ir.Array(bo, ir_dev)
    bc = ir.Array(bc, ir_dev)
    y = ir.Array(y, ir_dev)
    d_x = ir.Array(d_x, ir_dev)
    d_wf = ir.Array(d_wf, ir_dev)
    d_wi = ir.Array(d_wi, ir_dev)
    d_wo = ir.Array(d_wo, ir_dev)
    d_wc = ir.Array(d_wc, ir_dev)
    d_uf = ir.Array(d_uf, ir_dev)
    d_ui = ir.Array(d_ui, ir_dev)
    d_uo = ir.Array(d_uo, ir_dev)
    d_uc = ir.Array(d_uc, ir_dev)
    d_bf = ir.Array(d_bf, ir_dev)
    d_bi = ir.Array(d_bi, ir_dev)
    d_bo = ir.Array(d_bo, ir_dev)
    d_bc = ir.Array(d_bc, ir_dev)
    d_y = ir.Array(d_y, ir_dev)

    inference, forward = compile_all(in_feats, hidden_feats, length, ir_dev)
    warmup_num = 10
    test_num = 1000

    for i in range(warmup_num):
        inference(x, y, wi, ui, bi, wf, uf, bf, wc, uc, bc, wo, uo, bo)
        if i == 0:
            np.savetxt("y.out", y.numpy().reshape((hidden_feats, )))
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        inference(x, y, wi, ui, bi, wf, uf, bf, wc, uc, bc, wo, uo, bo)
    ir_dev.sync()
    t1 = time.time()

    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        forward(x, y, wi, ui, bi, wf, uf, bf, wc, uc, bc, wo, uo, bo)
    ir_dev.sync()
    t0 = time.time()
    for i in range(test_num):
        forward(x, y, wi, ui, bi, wf, uf, bf, wc, uc, bc, wo, uo, bo)
    ir_dev.sync()
    t1 = time.time()

    print(f"Forward Time = {(t1 - t0) / test_num * 1000} ms")

    # for i in range(warmup_num):
    #     backward(x, y, wi, ui, bi, wf, uf, bf, wc, uc, bc, wo, uo, bo,
    #                  d_wi, d_ui, d_bi, d_wf, d_uf, d_bf, d_wc, d_uc, d_bc, d_wo, d_uo, d_bo, d_x, d_y)
    #     if i == 0:
    #         np.savetxt("d_x.out", d_x.numpy().reshape((length, in_feats)))
    #         np.savetxt("d_wf.out", d_wf.numpy().reshape((hidden_feats, in_feats)))
    #         np.savetxt("d_wi.out", d_wi.numpy().reshape((hidden_feats, in_feats)))
    #         np.savetxt("d_wo.out", d_wo.numpy().reshape((hidden_feats, in_feats)))
    #         np.savetxt("d_wc.out", d_wc.numpy().reshape((hidden_feats, in_feats)))
    #         np.savetxt("d_uf.out", d_uf.numpy().reshape((hidden_feats, hidden_feats)))
    #         np.savetxt("d_ui.out", d_ui.numpy().reshape((hidden_feats, hidden_feats)))
    #         np.savetxt("d_uo.out", d_uo.numpy().reshape((hidden_feats, hidden_feats)))
    #         np.savetxt("d_uc.out", d_uc.numpy().reshape((hidden_feats, hidden_feats)))
    #         np.savetxt("d_bf.out", d_bf.numpy().reshape((hidden_feats, )))
    #         np.savetxt("d_bi.out", d_bi.numpy().reshape((hidden_feats, )))
    #         np.savetxt("d_bo.out", d_bo.numpy().reshape((hidden_feats, )))
    #         np.savetxt("d_bc.out", d_bc.numpy().reshape((hidden_feats, )))
    # ir_dev.sync()
    # t0 = time.time()
    # for i in range(test_num):
    #     backward(x, y, wi, ui, bi, wf, uf, bf, wc, uc, bc, wo, uo, bo,
    #                  d_wi, d_ui, d_bi, d_wf, d_uf, d_bf, d_wc, d_uc, d_bc, d_wo, d_uo, d_bo, d_x, d_y)
    # ir_dev.sync()
    # t1 = time.time()
    #
    # print(f"Backward Time = {(t1 - t0) / test_num * 1000} ms")
