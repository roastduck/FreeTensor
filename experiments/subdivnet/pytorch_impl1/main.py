import sys
import time
import numpy as np
import torch


def conv_impl1(adj, x, w0, w1, w2, w3):
    # TODO: Dilation
    # TODO: Stride
    # TODO: Batch

    n_faces = x.shape[0]
    in_feats = x.shape[1]
    out_feats = w0.shape[1]
    assert adj.shape == (n_faces, 3)
    assert x.shape == (n_faces, in_feats)
    assert w0.shape == (in_feats, out_feats)
    assert w1.shape == (in_feats, out_feats)
    assert w2.shape == (in_feats, out_feats)
    assert w3.shape == (in_feats, out_feats)

    adj_feat = torch.index_select(x, 0,
                                  adj.flatten()).reshape(n_faces, 3, in_feats)
    y0 = x @ w0
    y1 = torch.sum(adj_feat, dim=1) @ w1
    y2 = torch.sum(
        torch.abs(adj_feat -
                  torch.cat([adj_feat[:, 1:], adj_feat[:, :1]], dim=1)),
        dim=1) @ w2
    y3 = torch.sum(torch.abs(adj_feat - x.reshape(n_faces, 1, in_feats)),
                   dim=1) @ w3
    return y0 + y1 + y2 + y3


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu/gpu>")
        exit(-1)
    device = sys.argv[1]

    adj = torch.tensor(np.loadtxt("../adj.in", dtype=np.int32))
    n_faces = adj.shape[0]
    in_feats = 13
    out_feats = 64
    x = torch.tensor(np.loadtxt("../x.in"), dtype=torch.float)
    w0 = torch.tensor(np.loadtxt("../w0.in"), dtype=torch.float)
    w1 = torch.tensor(np.loadtxt("../w1.in"), dtype=torch.float)
    w2 = torch.tensor(np.loadtxt("../w2.in"), dtype=torch.float)
    w3 = torch.tensor(np.loadtxt("../w3.in"), dtype=torch.float)
    d_y = torch.tensor(np.loadtxt("../d_y.in"), dtype=torch.float)

    if device == 'gpu':
        adj = adj.cuda()
        x = x.cuda()
        w0 = w0.cuda()
        w1 = w1.cuda()
        w2 = w2.cuda()
        w3 = w3.cuda()
        d_y = d_y.cuda()
        sync = torch.cuda.synchronize
    else:
        assert device == 'cpu'
        sync = lambda: None

    warmup_num = 10
    test_num = 1000

    for i in range(warmup_num):
        y = conv_impl1(adj, x, w0, w1, w2, w3)
        if i == 0:
            np.savetxt("y.out", y.cpu().numpy())
    sync()
    t0 = time.time()
    for i in range(test_num):
        y = conv_impl1(adj, x, w0, w1, w2, w3)
    sync()
    t1 = time.time()
    assert y.shape == (n_faces, out_feats)
    print(f"Impl1 Inference Time = {(t1 - t0) / test_num * 1000} ms")

    x.requires_grad = True
    w0.requires_grad = True
    w1.requires_grad = True
    w2.requires_grad = True
    w3.requires_grad = True

    for i in range(warmup_num):
        y = conv_impl1(adj, x, w0, w1, w2, w3)
    sync()
    t0 = time.time()
    for i in range(test_num):
        y = conv_impl1(adj, x, w0, w1, w2, w3)
    sync()
    t1 = time.time()
    assert y.shape == (n_faces, out_feats)
    print(f"Impl1 Forward Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        y.backward(d_y, retain_graph=True)
        if i == 0:
            np.savetxt("d_x.out", x.grad.cpu().numpy())
            np.savetxt("d_w0.out", w0.grad.cpu().numpy())
            np.savetxt("d_w1.out", w1.grad.cpu().numpy())
            np.savetxt("d_w2.out", w2.grad.cpu().numpy())
            np.savetxt("d_w3.out", w3.grad.cpu().numpy())
    sync()
    t0 = time.time()
    for i in range(test_num):
        y.backward(d_y, retain_graph=True)
    sync()
    t1 = time.time()
    print(f"Impl1 Backward Time = {(t1 - t0) / test_num * 1000} ms")
