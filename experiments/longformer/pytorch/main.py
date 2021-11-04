import sys
import time
import math
import numpy as np
import torch


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
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu/gpu>")
        exit(-1)
    device = sys.argv[1]

    n_heads = 8
    seq_len = 10000
    feat_len = 512
    w = 32
    dilation = 4  # counts from 1
    dilation_heads = 2
    q = torch.rand(n_heads, seq_len, feat_len, dtype=torch.float)
    k = torch.rand(n_heads, seq_len, feat_len, dtype=torch.float)
    v = torch.rand(n_heads, seq_len, feat_len, dtype=torch.float)
    d_y = torch.rand(n_heads, seq_len, feat_len, dtype=torch.float)

    if device == 'gpu':
        q = q.cuda()
        k = k.cuda()
        v = v.cuda()
        d_y = d_y.cuda()
        sync = torch.cuda.synchronize
    else:
        assert device == 'cpu'
        sync = lambda: None

    warmup_num = 10
    test_num = 100

    for i in range(warmup_num):
        y = transformer_impl1(q, k, v, w, dilation, dilation_heads)
    sync()
    t0 = time.time()
    for i in range(test_num):
        y = transformer_impl1(q, k, v, w, dilation, dilation_heads)
    sync()
    t1 = time.time()
    assert y.shape == (n_heads, seq_len, feat_len)
    print(f"Impl1 Inference Time = {(t1 - t0) / test_num * 1000} ms")

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
    print(f"Impl1 Forward Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        y.backward(d_y, retain_graph=True)
    sync()
    t0 = time.time()
    for i in range(test_num):
        y.backward(d_y, retain_graph=True)
    sync()
    t1 = time.time()
    print(f"Impl2 Backward Time = {(t1 - t0) / test_num * 1000} ms")
