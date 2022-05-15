import freetensor as ft
import numpy as np
from freetensor.libop import *
import sys
sys.path.append('../..')
from common.numpy.io import load_txt, store_txt

target = ft.GPU()
device = ft.Device(target)


def test_subdivnet():
    a = 512
    b = 512
    m = 4
    # c = 64
    adj = load_txt("../adj.in", "int32")
    n_faces = adj.shape[0]
    in_feats = 13
    out_feats = 64

    @ft.transform
    def inference(adj, x, w0, w1, w2, w3, y):
        adj: ft.Var[(n_faces, 3), "int32", "input"]
        x: ft.Var[(n_faces, in_feats), "float32", "input"]
        w0: ft.Var[(in_feats, out_feats), "float32", "input"]
        w1: ft.Var[(in_feats, out_feats), "float32", "input"]
        w2: ft.Var[(in_feats, out_feats), "float32", "input"]
        w3: ft.Var[(in_feats, out_feats), "float32", "input"]
        y: ft.Var[(n_faces, out_feats), "float32", "output"]

        for i in range(n_faces):
            sum1 = zeros((in_feats,), "float32")
            sum2 = zeros((in_feats,), "float32")
            sum3 = zeros((in_feats,), "float32")
            for p in range(3):
                sum1[:] += x[adj[i, p]]
                sum2[:] += abs(x[adj[i, p]] - x[adj[i, (p + 1) % 3]])
                sum3[:] += abs(x[adj[i, p]] - x[i])
            y0 = matmul(x[i], w0)
            y1 = matmul(sum1, w1)
            y2 = matmul(sum2, w2)
            y3 = matmul(sum3, w3)
            y[i] = y0 + y1 + y2 + y3

    s = ft.Schedule(inference)
    print(s.ast())
    x = load_txt("../x.in", "float32")
    w0 = load_txt("../w0.in", "float32")
    w1 = load_txt("../w1.in", "float32")
    w2 = load_txt("../w2.in", "float32")
    w3 = load_txt("../w3.in", "float32")
    y = np.zeros((n_faces, out_feats), dtype="float32")
    d_x = np.zeros(x.shape, dtype='float32')
    d_w0 = np.zeros(w0.shape, dtype='float32')
    d_w1 = np.zeros(w1.shape, dtype='float32')
    d_w2 = np.zeros(w2.shape, dtype='float32')
    d_w3 = np.zeros(w3.shape, dtype='float32')
    d_y = load_txt("../d_y.in", "float32")
    ir_dev = device

    adj = ft.Array(adj, ir_dev)
    x = ft.Array(x, ir_dev)
    w0 = ft.Array(w0, ir_dev)
    w1 = ft.Array(w1, ir_dev)
    w2 = ft.Array(w2, ir_dev)
    w3 = ft.Array(w3, ir_dev)
    y = ft.Array(y, ir_dev)
    d_x = ft.Array(d_x, ir_dev)
    d_w0 = ft.Array(d_w0, ir_dev)
    d_w1 = ft.Array(d_w1, ir_dev)
    d_w2 = ft.Array(d_w2, ir_dev)
    d_w3 = ft.Array(d_w3, ir_dev)
    d_y = ft.Array(d_y, ir_dev)

    print("Start constructing...")
    s = ft.AutoSchedule(s, target, device, 128)
    s.set_params(adj=adj, x=x, w0=w0, w1=w1, w2=w2, w3=w3, y=y)
    # s.set_params(w=w_arr, x=x_arr, y=y_arr)
    print("Start running...")
    s = s.run(10)
    print("Start lowering...")
    func = ft.lower(s.func(), target)
    print(func)
    code = ft.codegen(func, target)
    print(code)


if __name__ == '__main__':
    test_subdivnet()
