import ir
import numpy as np

target = ir.GPU()
device = ir.Device(target)


def test_cache_read():
    a = 128
    b = 128

    @ir.transform
    def test(w, x, c, z):
        ir.declare_var(w, (a, b), "float32", "input", "gpu/global")
        ir.declare_var(x, (b, a), "float32", "input", "gpu/global")
        ir.declare_var(c, (b, a), "float32", "input", "gpu/global")
        y = ir.create_var((a, a), "float32", "gpu/local")
        ir.declare_var(z, (a, a), "float32", "output", "gpu/global")
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

    s = ir.Schedule(test)
    w_np = np.zeros((a, b), dtype="float32")
    x_np = np.zeros((b, a), dtype="float32")
    c_np = np.zeros((a, a), dtype="float32")
    z_np = np.zeros((a, a), dtype="float32")
    w_arr = ir.Array(w_np, device)
    x_arr = ir.Array(x_np, device)
    c_arr = ir.Array(c_np, device)
    z_arr = ir.Array(z_np, device)
    print("Start constructing...")
    s = ir.AutoSchedule(s, target, device, 128)
    s.set_params(w=w_arr, x=x_arr, c=c_arr, z=z_arr)
    sch = s.test_cache_read()
    func = ir.lower(sch.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)


test_cache_read()
