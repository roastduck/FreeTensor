import freetensor as ft
import numpy as np

target = ft.CPU()
device = ft.Device(target)


def test_multi_level_tiling():
    a = 128
    b = 256
    m = 4
    # c = 64

    @ft.transform
    # def test(w, x, y):
    def test(w, x, y, z, u):
        # def test(w, y):
        w: ft.Var((m, m, a, b), "int32", "input", "cpu")
        # w: ft.Var((a, b), "int32", "input", "cpu")
        x: ft.Var((m, m, b, a), "int32", "input", "cpu")
        # x: ft.Var((b, a), "int32", "input", "cpu")
        # y: ft.Var((a, b), "int32", "output", "cpu")
        y: ft.Var((m, m, a, a), "int32", "output", "cpu")
        # y: ft.Var((a, a), "int32", "output", "cpu")
        z: ft.Var((m, m, a, a), "int32", "output", "cpu")
        u: ft.Var((m, m), "int32", "output", "cpu")
        # "nid: L1"
        # for i in range(a):
        #     "nid: L2"
        #     for j in range(a):
        #         # for j in range(b):
        #         "nid: L3"
        #         for k in range(b):
        #             # for k in range(c):
        #             # y[i, j] = y[i, j] + w[i, j, k]
        #             y[i, j] = y[i, j] + w[i, k] * x[k, j]
        "nid: L1"
        for i in range(m):
            "nid: L2"
            for j in range(m):
                u[i, j] = i * j
                "nid: L3"
                for k in range(b):
                    "nid: L4"
                    for p in range(a):
                        "nid: L5"
                        for q in range(a):
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]
                "nid: L6"
                for p in range(a):
                    "nid: L7"
                    for q in range(a):
                        z[i, j, p, q] = y[i, j, p, q]

    s = ft.Schedule(test)
    # w_np = np.zeros((a, b, c), dtype="float32")
    # w_np = np.zeros((a, b), dtype="float32")
    # x_np = np.zeros((b, a), dtype="float32")
    # y_np = np.zeros((a, a), dtype="float32")
    w_np = np.zeros((m, m, a, b), dtype="float32")
    x_np = np.zeros((m, m, b, a), dtype="float32")
    y_np = np.zeros((m, m, a, a), dtype="float32")
    z_np = np.zeros((m, m, a, a), dtype="float32")
    u_np = np.zeros((m, m), dtype="float32")
    # y_np = np.zeros((a, b), dtype="float32")
    w_arr = ft.Array(w_np, device)
    x_arr = ft.Array(x_np, device)
    y_arr = ft.Array(y_np, device)
    z_arr = ft.Array(z_np, device)
    u_arr = ft.Array(u_np, device)
    print("Start constructing...")
    s = ft.AutoSchedule(s, target, device, 8)
    s.set_params(w=w_arr, x=x_arr, y=y_arr, z=z_arr, u=u_arr)
    # s.set_params(w=w_arr, x=x_arr, y=y_arr)
    print("Start running...")
    s = s.run(10)
    print("Start lowering...")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)


if __name__ == '__main__':
    test_multi_level_tiling()
