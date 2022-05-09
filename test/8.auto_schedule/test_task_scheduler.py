import freetensor as ft
import numpy as np

target = ft.GPU()
device = ft.Device(target)

a = 128
b = 128
m = 4


@ft.transform
def func1(w, x, y, z):
    w: ft.Var[(m, m, a, b), "int32", "input", "gpu/global"]
    x: ft.Var[(m, m, b, a), "int32", "input", "gpu/global"]
    y: ft.Var[(1, 1, a, a), "int32", "cache", "gpu/global"]
    z: ft.Var[(m, m, a, a), "int32", "output", "gpu/global"]
    "nid: L1"
    for i in range(m):
        "nid: L2"
        for j in range(m):
            "nid: L3"
            for k in range(b):
                "nid: L4"
                for p in range(a):
                    "nid: L5"
                    for q in range(a):
                        y[0, 0, p,
                          q] = y[0, 0, p, q] + w[i, j, p, k] * x[i, j, k, q]
            "nid: L6"
            for p in range(a):
                "nid: L7"
                for q in range(a):
                    z[i, j, p, q] = y[0, 0, p, q]


@ft.transform
def func2(w, x, c, z):
    w: ft.Var[(a, b), "float32", "input", "gpu/global"]
    x: ft.Var[(b, a), "float32", "input", "gpu/global"]
    c: ft.Var[(b, a), "float32", "input", "gpu/global"]
    y = ft.create_var((a, a), "float32", "gpu/local")
    z: ft.Var[(a, a), "float32", "output", "gpu/global"]
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


def test_task_scheduler():
    s1 = ft.Schedule(func1)
    s1 = ft.AutoSchedule(s1, target, device, 64)
    w_np = np.zeros((m, m, a, b), dtype="float32")
    x_np = np.zeros((m, m, b, a), dtype="float32")
    y_np = np.zeros((1, 1, a, a), dtype="float32")
    z_np = np.zeros((m, m, a, a), dtype="float32")
    w_arr = ft.Array(w_np, device)
    x_arr = ft.Array(x_np, device)
    y_arr = ft.Array(y_np, device)
    z_arr = ft.Array(z_np, device)
    s1.set_params(w=w_arr, x=x_arr, y=y_arr, z=z_arr)
    s2 = ft.Schedule(func2)
    s2 = ft.AutoSchedule(s2, target, device, 64)
    w_np = np.zeros((a, b), dtype="float32")
    x_np = np.zeros((b, a), dtype="float32")
    c_np = np.zeros((a, a), dtype="float32")
    z_np = np.zeros((a, a), dtype="float32")
    w_arr = ft.Array(w_np, device)
    x_arr = ft.Array(x_np, device)
    c_arr = ft.Array(c_np, device)
    z_arr = ft.Array(z_np, device)
    s2.set_params(w=w_arr, x=x_arr, c=c_arr, z=z_arr)

    ts = ft.TaskScheduler([s1, s2], measures_per_round=32)
    ts.tune(20)


test_task_scheduler()
