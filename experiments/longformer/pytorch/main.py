import sys
import time
import math
import argparse
import numpy as np
import torch

sys.path.append('../..')
from common.numpy.io import load_txt, store_txt


def dilated_attention(q, k, v, w, dilation):
    n_heads, seq_len, feat_len = q.shape
    assert q.shape == (n_heads, seq_len, feat_len)
    assert k.shape == (n_heads, seq_len, feat_len)
    assert v.shape == (n_heads, seq_len, feat_len)

    sqrt_d = math.sqrt(feat_len)

    pad_k = torch.nn.functional.pad(k, (0, 0, w * dilation, w * dilation))
    pad_v = torch.nn.functional.pad(v, (0, 0, w * dilation, w * dilation))
    assert pad_k.shape == (n_heads, seq_len + 2 * w * dilation, feat_len)
    assert pad_v.shape == (n_heads, seq_len + 2 * w * dilation, feat_len)
    diag_k = pad_k.as_strided(size=(n_heads, seq_len, 2 * w + 1, feat_len),
                              stride=((seq_len + 2 * w * dilation) * feat_len,
                                      feat_len, feat_len * dilation, 1))
    diag_v = pad_v.as_strided(size=(n_heads, seq_len, 2 * w + 1, feat_len),
                              stride=((seq_len + 2 * w * dilation) * feat_len,
                                      feat_len, feat_len * dilation, 1))

    attn = torch.einsum("ijp,ijkp->ijk", q, diag_k)
    assert attn.shape == (n_heads, seq_len, 2 * w + 1)
    attn = torch.nn.functional.softmax(attn, dim=-1) / sqrt_d

    return torch.einsum("ijk,ijkp->ijp", attn, diag_v)


def transformer_impl1(q, k, v, w, dilation, dilation_heads):
    front_heads = dilated_attention(q[:dilation_heads], k[:dilation_heads],
                                    v[:dilation_heads], w, dilation)
    back_heads = dilated_attention(q[dilation_heads:], k[dilation_heads:],
                                   v[dilation_heads:], w, 1)
    return torch.cat([front_heads, back_heads], dim=0)


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

    n_heads = 8
    seq_len = 10000
    feat_len = 512
    w = 32
    dilation = 4  # counts from 1
    dilation_heads = 2
    q = torch.tensor(load_txt("../q.in", "float32"), dtype=torch.float)
    k = torch.tensor(load_txt("../k.in", "float32"), dtype=torch.float)
    v = torch.tensor(load_txt("../v.in", "float32"), dtype=torch.float)
    d_y = torch.tensor(load_txt("../d_y.in", "float32"), dtype=torch.float)

    if device == 'gpu':
        q = q.cuda()
        k = k.cuda()
        v = v.cuda()
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
        y = transformer_impl1(q, k, v, w, dilation, dilation_heads)
        if i == 0:
            store_txt("y.out", y.cpu().numpy())
    sync()
    if cmd_args.profile_gpu:
        profile_start()
    t0 = time.time()
    for i in range(test_num):
        y = transformer_impl1(q, k, v, w, dilation, dilation_heads)
    sync()
    t1 = time.time()
    if cmd_args.profile_gpu:
        profile_stop()
    assert y.shape == (n_heads, seq_len, feat_len)
    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")

    if cmd_args.profile_gpu:
        exit(0)

    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    for i in range(warmup_num):
        y = transformer_impl1(q, k, v, w, dilation, dilation_heads)
    sync()
    t0 = time.time()
    for i in range(test_num):
        y = transformer_impl1(q, k, v, w, dilation, dilation_heads)
    sync()
    t1 = time.time()
    assert y.shape == (n_heads, seq_len, feat_len)
    print(f"Forward Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        y.backward(d_y, retain_graph=True)
        if i == 0:
            store_txt("d_q.out", q.grad.cpu().numpy())
            store_txt("d_k.out", k.grad.cpu().numpy())
            store_txt("d_v.out", v.grad.cpu().numpy())
    sync()
    t0 = time.time()
    for i in range(test_num):
        y.backward(d_y, retain_graph=True)
    sync()
    t1 = time.time()
    print(f"Backward Time = {(t1 - t0) / test_num * 1000} ms")
