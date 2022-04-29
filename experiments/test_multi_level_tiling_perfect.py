import freetensor as ft
import numpy as np

target = ft.GPU()
device = ft.Device(target)


def test_multi_level_tiling():
    a = 1024
    b = 1024
    m = 4
    # c = 64

    @ft.transform
    # def test(w, x, y):
    def test(w, x, c, z):
        # def test(w, y):
        ft.declare_var(w, (a, b), "float32", "input", "gpu/global")
        # ft.declare_var(w, (a, b), "int32", "input", "cpu")
        ft.declare_var(x, (b, a), "float32", "input", "gpu/global")
        ft.declare_var(c, (b, a), "float32", "input", "gpu/global")
        # ft.declare_var(x, (b, a), "int32", "input", "cpu")
        # ft.declare_var(y, (a, b), "int32", "output", "cpu")
        y = ft.create_var((a, a), "float32", "gpu/local")
        # ft.declare_var(y, (a, a), "int32", "output", "cpu")
        ft.declare_var(z, (a, a), "float32", "output", "gpu/global")
        # ft.declare_var(u, (m, m), "int32", "output", "gpu/global")
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
        "nid: L3"
        for k in range(b):
            "nid: L4"
            for p in range(a):
                "nid: L5"
                for q in range(a):
                    y[p, q] = y[p, q] + w[p, k] * x[k, q]
        "nid: L6"
        for p in range(a):
            "nid: L7"
            for q in range(a):
                z[p, q] = y[p, q] + c[p, q]

    s = ft.Schedule(test)
    s.cache("L3", "w", "gpu/shared")
    s.cache("L3", "x", "gpu/shared")
    # w_np = np.zeros((a, b, c), dtype="float32")
    # w_np = np.zeros((a, b), dtype="float32")
    # x_np = np.zeros((b, a), dtype="float32")
    # y_np = np.zeros((a, a), dtype="float32")
    w_np = np.zeros((a, b), dtype="float32")
    x_np = np.zeros((b, a), dtype="float32")
    c_np = np.zeros((a, a), dtype="float32")
    z_np = np.zeros((a, a), dtype="float32")
    # u_np = np.zeros((m, m), dtype="float32")
    # y_np = np.zeros((a, b), dtype="float32")
    w_arr = ft.Array(w_np, device)
    x_arr = ft.Array(x_np, device)
    c_arr = ft.Array(c_np, device)
    z_arr = ft.Array(z_np, device)
    # u_arr = ft.Array(u_np, device)
    print("Start constructing...")
    s = ft.AutoSchedule(s, target, device, 128)
    s.set_params(w=w_arr, x=x_arr, c=c_arr, z=z_arr)
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
