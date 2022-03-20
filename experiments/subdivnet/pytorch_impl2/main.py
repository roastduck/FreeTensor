import sys
import time
import argparse
import numpy as np
import torch

sys.path.append('../..')
from common.numpy.io import load_txt, store_txt


def conv_impl2(adj, x, w0, w1, w2, w3):
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
    sum1 = torch.zeros_like(x)
    sum2 = torch.zeros_like(x)
    sum3 = torch.zeros_like(x)
    for p in range(3):
        sum1 += adj_feat[:, p]
        sum2 += torch.abs(adj_feat[:, p] - adj_feat[:, (p + 1) % 3])
        sum3 += torch.abs(adj_feat[:, p] - x)

    return x @ w0 + sum1 @ w1 + sum2 @ w2 + sum3 @ w3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target', nargs='?')
    parser.add_argument('--warmup-repeat',
                        type=int,
                        default=10,
                        dest='warmup_num')
    parser.add_argument('--timing-repeat',
                        type=int,
                        default=100,
                        dest='test_num')
    parser.add_argument('--profile-gpu',
                        action='store_true',
                        dest='profile_gpu')
    cmd_args = parser.parse_args()

    if cmd_args.profile_gpu:
        from common.gpu import profile_start, profile_stop

    device = cmd_args.target

    adj = torch.tensor(load_txt("../adj.in", "int32"))
    n_faces = adj.shape[0]
    in_feats = 13
    out_feats = 64
    x = torch.tensor(load_txt("../x.in", "float32"), dtype=torch.float)
    w0 = torch.tensor(load_txt("../w0.in", "float32"), dtype=torch.float)
    w1 = torch.tensor(load_txt("../w1.in", "float32"), dtype=torch.float)
    w2 = torch.tensor(load_txt("../w2.in", "float32"), dtype=torch.float)
    w3 = torch.tensor(load_txt("../w3.in", "float32"), dtype=torch.float)
    d_y = torch.tensor(load_txt("../d_y.in", "float32"), dtype=torch.float)

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

    print(
        f"{cmd_args.warmup_num} warmup, {cmd_args.test_num} repeats for evalution"
    )
    warmup_num = cmd_args.warmup_num
    test_num = cmd_args.test_num

    for i in range(warmup_num):
        y = conv_impl2(adj, x, w0, w1, w2, w3)
        if i == 0:
            store_txt("y.out", y.cpu().numpy())
    sync()
    if cmd_args.profile_gpu:
        profile_start()
    t0 = time.time()
    for i in range(test_num):
        y = conv_impl2(adj, x, w0, w1, w2, w3)
    sync()
    t1 = time.time()
    if cmd_args.profile_gpu:
        profile_stop()
    assert y.shape == (n_faces, out_feats)
    print(f"Impl2 Inference Time = {(t1 - t0) / test_num * 1000} ms")

    if cmd_args.profile_gpu:
        exit(0)

    x.requires_grad = True
    w0.requires_grad = True
    w1.requires_grad = True
    w2.requires_grad = True
    w3.requires_grad = True

    for i in range(warmup_num):
        y = conv_impl2(adj, x, w0, w1, w2, w3)
    sync()
    t0 = time.time()
    for i in range(test_num):
        y = conv_impl2(adj, x, w0, w1, w2, w3)
    sync()
    t1 = time.time()
    assert y.shape == (n_faces, out_feats)
    print(f"Impl2 Forward Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        y.backward(d_y, retain_graph=True)
        if i == 0:
            store_txt("d_x.out", x.grad.cpu().numpy())
            store_txt("d_w0.out", w0.grad.cpu().numpy())
            store_txt("d_w1.out", w1.grad.cpu().numpy())
            store_txt("d_w2.out", w2.grad.cpu().numpy())
            store_txt("d_w3.out", w3.grad.cpu().numpy())
    sync()
    t0 = time.time()
    for i in range(test_num):
        y.backward(d_y, retain_graph=True)
    sync()
    t1 = time.time()
    print(f"Impl2 Backward Time = {(t1 - t0) / test_num * 1000} ms")
