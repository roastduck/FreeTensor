import sys
import time
import math
import numpy as np
import ir
import ir.debug
from ir.libop import StaticType as T

jit_cache = {}


def conv(x, w1, w2, n, c_in, c_out, h, w, k_h, k_w, device, mtype, local_mtype):
    if (x, w1, w2, n, c_in, c_out, h, w, k_h, k_w) in jit_cache:
        exe = jit_cache[(x, w1, w2, n, c_in, c_out, h, w, k_h, k_w)]

    else:

        # yapf: disable

        @ir.transform
        def f(X, W1, W2, Y):
            ir.declare_var(X, (n, c_in, h, w), "float32", "input", mtype)
            ir.declare_var(W1, (k_h, k_w, 2, c_in, k_h, k_w), "float32", "input", mtype)
            ir.declare_var(W2, (c_out, c_in, k_h, k_w), "float32", "input", mtype)
            ir.declare_var(Y, (n, c_out, h, w), "float32", "output", mtype)

            "nid: Li"
            for i in range(n):
                "nid: Lp"
                for p in range(h):
                    "nid: Lq"
                    for q in range(w):
                        row = ir.create_var((k_h, k_w), "float32", "cache", local_mtype)
                        col = ir.create_var((k_h, k_w), "float32", "cache", local_mtype)
                        row_int = ir.create_var((k_h, k_w), "int32", "cache", local_mtype)
                        col_int = ir.create_var((k_h, k_w), "int32", "cache", local_mtype)
                        for ro in range(k_h):
                            for so in range(k_w):
                                row[ro, so] = 0
                                col[ro, so] = 0
                                for ki in range(c_in):
                                    "nid: Lri"
                                    for ri in range(k_h):
                                        "nid: Lsi"
                                        for si in range(k_w):
                                            if p + ri >= 0 and p + ri < h and q + si >= 0 and q + si < w:
                                                row[ro, so] += X[i, ki, p + ri, q + si] * W1[ro, so, 0, ki, ri, si]
                                                col[ro, so] += X[i, ki, p + ri, q + si] * W1[ro, so, 1, ki, ri, si]
                                row[ro, so] /= c_in
                                col[ro, so] /= c_in
                                row_int[ro, so] = ir.cast(ir.floor(row[ro, so]), "int32")
                                col_int[ro, so] = ir.cast(ir.floor(col[ro, so]), "int32")

                        pixel = ir.create_var((c_in, k_h, k_w), "float32", "cache", local_mtype)
                        for ki in range(c_in):
                            "nid: Lro1"
                            for ro in range(k_h):
                                "nid: Lso1"
                                for so in range(k_w):
                                    x = ir.create_var((), "int32", "cache", local_mtype)
                                    y = ir.create_var((), "int32", "cache", local_mtype)
                                    x[()] = p + ro + row_int[ro, so]
                                    y[()] = q + so + col_int[ro, so]
                                    pixel[ki, ro, so] = 0
                                    if x[()] >= 0 and x[()] < h and y[()] >= 0 and y[()] < w:
                                        pixel[ki, ro, so] += X[i, ki, x[()], y[()]] * (
                                                row[ro, so] - row_int[ro, so]) * (
                                                        col[ro, so] - col_int[ro, so])
                                    if x[()] >= 0 and x[()] < h and y[()] + 1 >= 0 and y[()] + 1 < w:
                                        pixel[ki, ro, so] += X[i, ki, x[()], y[()] + 1] * (
                                                row[ro, so] - row_int[ro, so]) * (
                                                        col_int[ro, so] + 1 - col[ro, so])
                                    if x[()] + 1 >= 0 and x[()] + 1 < h and y[()] >= 0 and y[()] < w:
                                        pixel[ki, ro, so] += X[i, ki, x[()] + 1, y[()]] * (
                                                row_int[ro, so] + 1 - row[ro, so]) * (
                                                        col[ro, so] - col_int[ro, so])
                                    if x[()] + 1 >= 0 and x[()] + 1 < h and y[()] + 1 >= 0 and y[()] + 1 < w:
                                        pixel[ki, ro, so] += X[i, ki, x[()] + 1, y[()] + 1] * (
                                                row_int[ro, so] + 1 - row[ro, so]) * (
                                                        col_int[ro, so] + 1 - col[ro, so])

                        "nid: Lko"
                        for ko in range(c_out):
                            "nid: init_Y"
                            Y[i, ko, p, q] = 0
                            "nid: Lki3"
                            for ki in range(c_in):
                                "nid: Lro2"
                                for ro in range(k_h):
                                    "nid: Lso2"
                                    for so in range(k_w):
                                        Y[i, ko, p, q] += pixel[ki, ro, so] * W2[ko, ki, ro, so]

        # yapf: enable

        s = ir.Schedule(f)
        print(s.ast())
        if device.target().type() == ir.TargetType.CPU:
            Lko = s.move_to("Lko", ir.MoveToSide.After, "Li")
            _, flush, Y_t = s.cache(Lko, "Y", "cpu")
            Y_t_def = s.find(lambda x: x.node_type() == ir.ASTNodeType.VarDef
                             and x.node().name == Y_t)
            s.var_reorder(Y_t_def, [0, 2, 3, 1])
            flush_loop = s.find(lambda x: x.nid() == flush).outer().outer(
            ).outer().outer().node()
            s.parallelize(flush_loop, 'openmp')
            s.parallelize(flush_loop.body, 'openmp')
            s.as_matmul(Lko)
            Lipq = s.merge(s.merge('Li', 'Lp'), 'Lq')
            s.parallelize(Lipq, 'openmp')
            s.unroll("Lri")
            s.unroll("Lsi")
            s.unroll("Lro1")
            s.unroll("Lso1")
        else:
            Lipq = s.merge(s.merge('Li', 'Lp'), 'Lq')
            blk, thr = s.split(Lipq, 128)
            s.parallelize(blk, 'blockIdx.x')
            s.parallelize(thr, 'threadIdx.x')
            s.cache("Lko", "Y", "gpu/local")
        f = ir.lower(s.func(), device.target())
        print(f)
        code = ir.codegen(f, device.target())
        print(ir.debug.with_line_no(code))
        exe = ir.Driver(f, code, device)
        exe.set_params(x, w1, w2, y)
        # TODO: do not set_params here
        jit_cache[(x, w1, w2, n, c_in, c_out, h, w, k_h, k_w)] = exe

    exe.run()
    exe.sync()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu/gpu>")
        exit(-1)
    device = sys.argv[1]

    n = 8
    c_in = 256
    c_out = 256
    h = 56
    w = 56
    k_h = 3
    k_w = 3
    x = np.random.uniform(size=(n, c_in, h, w)).astype("float32") * 2 - 1
    w1 = np.random.uniform(size=(k_h, k_w, 2, c_in, k_h,
                                 k_w)).astype("float32") * 2 - 1
    w2 = np.random.uniform(size=(c_out, c_in, k_h,
                                 k_w)).astype("float32") * 2 - 1
    y = np.zeros((n, c_out, h, w), dtype="float32")

    if device == 'gpu':
        ir_dev = ir.Device(ir.GPU())
        ir_mtype = 'gpu/global'
        ir_local_mtype = 'gpu/local'
    else:
        assert device == 'cpu'
        ir_dev = ir.Device(ir.CPU())
        ir_mtype = 'cpu'
        ir_local_mtype = 'cpu'

    x = ir.Array(x, ir_dev)
    w1 = ir.Array(w1, ir_dev)
    w2 = ir.Array(w2, ir_dev)
    y = ir.Array(y, ir_dev)

    test_num = 100
    conv(x, w1, w2, n, c_in, c_out, h, w, k_h, k_w, ir_dev, ir_mtype,
         ir_local_mtype)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        conv(x, w1, w2, n, c_in, c_out, h, w, k_h, k_w, ir_dev, ir_mtype,
             ir_local_mtype)
    t1 = time.time()

    print(f"Time = {(t1 - t0) / test_num * 1000} ms")
