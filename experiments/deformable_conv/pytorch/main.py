import sys
import time
import math
import numpy as np
import torch


def conv_impl1(x, w1, w2):
    n, c_in, h, w = x.shape
    c_out = w2.shape[0]
    k_h = w1.shape[0]
    k_w = w1.shape[1]
    assert x.shape == (n, c_in, h, w)
    assert w1.shape == (k_h, k_w, 2, c_in, k_h, k_w)
    assert w2.shape == (c_out, c_in, k_h, k_w)

    offset = torch.conv2d(x,
                          w1.reshape(-1, c_in, k_h, k_w),
                          padding=(k_h // 2,
                                   k_w // 2)).reshape(n, k_h, k_w, 2, h, w)
    offset /= c_in  # Make offset not too large
    offset = offset.permute(0, 4, 5, 1, 2, 3)
    assert offset.shape == (n, h, w, k_h, k_w, 2)

    offset += torch.stack(torch.meshgrid(
        torch.arange(-(k_h // 2), k_h - k_h // 2, device=x.device),
        torch.arange(-(k_w // 2), k_w - k_w // 2, device=x.device)),
                          dim=-1).reshape(k_h, k_w, 2)

    coords = torch.stack(torch.meshgrid(torch.arange(h, device=x.device),
                                        torch.arange(w, device=x.device)),
                         dim=-1).reshape(h, w, 1, 1, 2)
    coords = coords + offset
    assert coords.shape == (n, h, w, k_h, k_w, 2)

    coords = coords.reshape(n, 1, h, w, k_h, k_w, 2)
    coords_int = torch.floor(coords).to(torch.long)
    # floor is necessary because floor(-1.5) = -2
    # tensors used as indices must be long, byte or bool tensors

    # We apply an 1-pixel wide padding, and limit the coordinate to not exceeding the padding
    # Rational:
    # 1. torch.where computes both cases and then select, which is infeasible because of out-of-range array accesses
    # 2. A dynamic-sized padding is suboptimal
    x_pad = torch.nn.functional.pad(x, (1, 1, 1, 1))
    assert x_pad.shape == (n, c_in, h + 2, w + 2)
    x_or_pad = lambda sample, row, col: x_pad[sample, :,
                                              torch.clamp(row + 1, 0, h + 1),
                                              torch.clamp(col + 1, 0, w + 1)]

    p00 = x_or_pad(torch.repeat_interleave(torch.arange(n), h * w * k_h * k_w),
                   coords_int.reshape(-1, 2)[:, 0],
                   coords_int.reshape(-1, 2)[:, 1]).reshape(
                       n, h, w, k_h, k_w, c_in).permute(0, 5, 1, 2, 3, 4)
    p01 = x_or_pad(torch.repeat_interleave(torch.arange(n), h * w * k_h * k_w),
                   coords_int.reshape(-1, 2)[:, 0],
                   coords_int.reshape(-1, 2)[:, 1] + 1).reshape(
                       n, h, w, k_h, k_w, c_in).permute(0, 5, 1, 2, 3, 4)
    p10 = x_or_pad(torch.repeat_interleave(torch.arange(n), h * w * k_h * k_w),
                   coords_int.reshape(-1, 2)[:, 0] + 1,
                   coords_int.reshape(-1, 2)[:, 1]).reshape(
                       n, h, w, k_h, k_w, c_in).permute(0, 5, 1, 2, 3, 4)
    p11 = x_or_pad(torch.repeat_interleave(torch.arange(n), h * w * k_h * k_w),
                   coords_int.reshape(-1, 2)[:, 0] + 1,
                   coords_int.reshape(-1, 2)[:, 1] + 1).reshape(
                       n, h, w, k_h, k_w, c_in).permute(0, 5, 1, 2, 3, 4)

    pixels = p00 * (coords.select(-1, 0) - coords_int.select(-1, 0)) * (
        coords.select(-1, 1) - coords_int.select(-1, 1))
    pixels += p01 * (coords.select(-1, 0) - coords_int.select(-1, 0)) * (
        coords_int.select(-1, 1) + 1 - coords.select(-1, 1))
    pixels += p10 * (coords_int.select(-1, 0) + 1 - coords.select(-1, 0)) * (
        coords.select(-1, 1) - coords_int.select(-1, 1))
    pixels += p11 * (coords_int.select(-1, 0) + 1 - coords.select(-1, 0)) * (
        coords_int.select(-1, 1) + 1 - coords.select(-1, 1))
    assert pixels.shape == (n, c_in, h, w, k_h, k_w)

    return torch.einsum("nchwrs,kcrs->nkhw", pixels, w2)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu/gpu>")
        exit(-1)
    device = sys.argv[1]

    n = 8
    c_in = 256
    c_out = 256
    h = 56
    w = 56
    k_h = 3
    k_w = 3

    x = torch.rand(n, c_in, h, w, dtype=torch.float) * 2 - 1
    w1 = torch.rand(k_h, k_w, 2, c_in, k_h, k_w, dtype=torch.float) * 2 - 1
    w2 = torch.rand(c_out, c_in, k_h, k_w, dtype=torch.float) * 2 - 1

    if device == 'gpu':
        x = x.cuda()
        w1 = w1.cuda()
        w2 = w2.cuda()
    else:
        assert device == 'cpu'

    test_num = 100

    y = conv_impl1(x, w1, w2)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        y = conv_impl1(x, w1, w2)
    t1 = time.time()
    assert y.shape == (n, c_out, h, w)
    print(f"Impl1 Time = {(t1 - t0) / test_num * 1000} ms")
