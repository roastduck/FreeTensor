import freetensor as ft
import numpy as np

target = ft.GPU()
device = ft.Device(target)


def test_multi_level_tiling():
    a = 128
    b = 128
    m = 4
    # c = 64

    @ft.transform
    # def test(w, x, y):
    def test(w, x, c, z):
        # def test(w, y):
        w: ft.Var[(a, b), "float32", "input", "gpu/global"]
        # w: ft.Var[(a, b), "int32", "input", "cpu"]
        x: ft.Var[(b, a), "float32", "input", "gpu/global"]
        c: ft.Var[(b, a), "float32", "input", "gpu/global"]
        # x: ft.Var[(b, a), "int32", "input", "cpu"]
        # y: ft.Var[(a, b), "int32", "output", "cpu"]
        y = ft.empty((a, a), "float32", "gpu/local")
        # y: ft.Var[(a, a), "int32", "output", "cpu"]
        z: ft.Var[(a, a), "float32", "output", "gpu/global"]
        # u: ft.Var[(m, m), "int32", "output", "gpu/global"]
        # #! nid: L1
        # for i in range(a):
        #     #! nid: L2
        #     for j in range(a):
        #         # for j in range(b):
        #         #! nid: L3
        #         for k in range(b):
        #             # for k in range(c):
        #             # y[i, j] = y[i, j] + w[i, j, k]
        #             y[i, j] = y[i, j] + w[i, k] * x[k, j]
        #! nid: L4
        for p in range(a):
            #! nid: L5
            for q in range(a):
                y[p, q] = 0
                #! nid: L3
                for k in range(b):
                    y[p, q] = y[p, q] + w[p, k] * x[k, q]
        #! nid: L6
        for p in range(a):
            #! nid: L7
            for q in range(a):
                z[p, q] = y[p, q] + c[p, q]

    s = ft.Schedule(test)
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
