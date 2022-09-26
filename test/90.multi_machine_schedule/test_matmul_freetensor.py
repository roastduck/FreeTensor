import freetensor as ft
import numpy as np
import pytest
import sys


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_matmul():
    try:
        device = ft.GPU()
        target = device.target()
    except Exception:
        sys.exit(1)
    NPORT = 27227
    tmp = ft.RPCTool(host="None",
                     self_server_ip="127.0.0.1",
                     self_server_port=NPORT,
                     sev_status=[])
    tmp.server_auto_shutdown(10)
    client = ft.MultiMachineScheduler(addr="127.0.0.1", port=NPORT)
    a = 256
    b = 256
    m = 4
    # c = 64

    @ft.transform
    # def test(w, x, y):
    def test(w, x, c, z):
        # def test(w, y):
        w: ft.Var[(a, b), "float32", "input", "gpu/global"]
        # w: ft.Var[(a, b), "int32", "input", "cpu"]
        x: ft.Var[(b, a), "float32", "input", "gpu/global"]
        c: ft.Var[(a, a), "float32", "input", "gpu/global"]
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
    w_arr = ft.Array(w_np)
    x_arr = ft.Array(x_np)
    c_arr = ft.Array(c_np)
    z_arr = ft.Array(z_np)
    # u_arr = ft.Array(u_np, device)
    print("Start constructing...")
    s = ft.AutoSchedule(s,
                        target,
                        device,
                        tag="matmul",
                        min_block_size=256,
                        verbose=2,
                        remote_measure_submit=client.remote_measure_submit)
    s.set_params(w=w_arr, x=x_arr, c=c_arr, z=z_arr)
    # s.set_params(w=w_arr, x=x_arr, y=y_arr)
    print("Start running...")
    s = s.run(1)
    print("Start lowering...")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)

    client.rpctool.end_server()
