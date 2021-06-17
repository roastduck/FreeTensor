import ir
import numpy as np

target = ir.CPU()
device = ir.Device(target)


def test_multi_level_tiling():
    a = 128
    b = 256
    @ir.transform
    def test(w, x, y):
        # ir.declare_var(n, (1,), "int32", "input", "byvalue")
        # ir.declare_var(m, (1,), "int32", "input", "byvalue")
        ir.declare_var(w, (a, b), "int32", "input", "cpu")
        ir.declare_var(x, (b, a), "int32", "input", "cpu")
        ir.declare_var(y, (a, a), "int32", "output", "cpu")
        # ir.declare_var(z, (128, 128), "int32", "output", "cpu")
        "nid: L0"
        for l in range(2):
            "nid: L1"
            for i in range(a):
                "nid: L2"
                for j in range(a):
                    "nid: L3"
                    for k in range(b):
                        y[i, j] = y[i, j] + w[i, k] * x[k, j]
                    # "nid: L4"
                    # for k in range(0, 256):
                    #     z[i, j] = w[i, k] + x[k, j]
    s = ir.Schedule(test)
    w_np = np.zeros((a, b), dtype="float32")
    x_np = np.zeros((b, a), dtype="float32")
    y_np = np.zeros((a, a), dtype="float32")
    w_arr = ir.Array(w_np, device)
    x_arr = ir.Array(x_np, device)
    y_arr = ir.Array(y_np, device)
    s = ir.AutoSchedule(s, target, device)
    s.set_params(w=w_arr, x=x_arr, y=y_arr)
    s = s.run(500, 30, 60)
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)
