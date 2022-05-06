import sys
import time
import math
import numpy as np
import freetensor as ft
from freetensor import debug

jit_cache = {}


def conv(x, w1, w2, y, n, c_in, c_out, h, w, k_h, k_w, device):
    if (n, c_in, c_out, h, w, k_h, k_w) in jit_cache:
        exe = jit_cache[(n, c_in, c_out, h, w, k_h, k_w)]

    else:
        mtype = device.main_mem_type()

        # yapf: disable

        @ft.transform
        def f(X, W1, W2, Y):
            X: ft.Var[(n, c_in, h, w), "float32", "input", mtype]
            W1: ft.Var[(k_h, k_w, 2, c_in, k_h, k_w), "float32", "input", mtype]
            W2: ft.Var[(c_out, c_in, k_h, k_w), "float32", "input", mtype]
            Y: ft.Var[(n, c_out, h, w), "float32", "output", mtype]

            "nid: Li"
            for i in range(n):
                "nid: Lp"
                for p in range(h):
                    "nid: Lq"
                    for q in range(w):
                        row = ft.empty((k_h, k_w), "float32", mtype)
                        col = ft.empty((k_h, k_w), "float32", mtype)
                        row_int = ft.empty((k_h, k_w), "int32", mtype)
                        col_int = ft.empty((k_h, k_w), "int32", mtype)
                        "nid: Lro0"
                        for ro in range(k_h):
                            "nid: Lso0"
                            for so in range(k_w):
                                row[ro, so] = 0
                                col[ro, so] = 0
                                "nid: Lki0"
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
                                row_int[ro, so] = ft.cast(ft.floor(row[ro, so]), "int32")
                                col_int[ro, so] = ft.cast(ft.floor(col[ro, so]), "int32")

                        pixel = ft.empty((c_in, k_h, k_w), "float32", mtype)
                        "nid: Lki1"
                        for ki in range(c_in):
                            "nid: Lro1"
                            for ro in range(k_h):
                                "nid: Lso1"
                                for so in range(k_w):
                                    x = ft.empty((), "int32", mtype)
                                    y = ft.empty((), "int32", mtype)
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
                            Y[i, ko, p, q] = 0
                            for ki in range(c_in):
                                for ro in range(k_h):
                                    for so in range(k_w):
                                        Y[i, ko, p, q] += pixel[ki, ro, so] * W2[ko, ki, ro, so]

        # yapf: enable

        s = ft.Schedule(f)
        print(s.ast())
        if device.target().type() == ft.TargetType.CPU:
            Lko = s.move_to("Lko", ft.MoveToSide.After, "Li")
            _, _, _, Y_t_def = s.cache(Lko, "Y", "cpu")
            s.var_reorder(Y_t_def, [0, 2, 3, 1])
            s.auto_schedule(device.target())
        else:
            Lko = s.move_to("Lko", ft.MoveToSide.After, "Li")
            _, _, _, Y_t_def = s.cache(Lko, "Y", "gpu/global")
            s.var_reorder(Y_t_def, [0, 2, 3, 1])
            s.var_reorder(":pixel", [3, 4, 5, 0, 1, 2])
            s.auto_schedule(device.target())
        f = ft.lower(s.func(), device.target())
        print(f)
        code = ft.codegen(f, device.target())
        print(debug.with_line_no(code))
        exe = ft.Driver(f, code, device)
        exe.set_params(x, w1, w2, y)
        # TODO: do not set_params here
        jit_cache[(n, c_in, c_out, h, w, k_h, k_w)] = exe

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
        ir_dev = ft.Device(ft.GPU())
        ir_mtype = 'gpu/global'
        ir_local_mtype = 'gpu/local'
    else:
        assert device == 'cpu'
        ir_dev = ft.Device(ft.CPU())
        ir_mtype = 'cpu'
        ir_local_mtype = 'cpu'

    x = ft.Array(x, ir_dev)
    w1 = ft.Array(w1, ir_dev)
    w2 = ft.Array(w2, ir_dev)
    y = ft.Array(y, ir_dev)

    test_num = 100
    conv(x, w1, w2, y, n, c_in, c_out, h, w, k_h, k_w, ir_dev)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        conv(x, w1, w2, y, n, c_in, c_out, h, w, k_h, k_w, ir_dev)
    t1 = time.time()

    print(f"Time = {(t1 - t0) / test_num * 1000} ms")
