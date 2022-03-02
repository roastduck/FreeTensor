import sys
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp

sys.path.append('../..')
from common.jax.io import load_txt, store_txt


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

    sum1 = jnp.zeros_like(x)
    sum2 = jnp.zeros_like(x)
    sum3 = jnp.zeros_like(x)
    for p in range(3):
        adj_p_feat = jax.vmap(lambda f3: x[f3[p]])(adj)
        adj_p_plus_1_feat = jax.vmap(lambda f3: x[f3[(p + 1) % 3]])(adj)
        sum1 += adj_p_feat
        sum2 += jnp.abs(adj_p_feat - adj_p_plus_1_feat)
        sum3 += jnp.abs(adj_p_feat - x)

    return x @ w0 + sum1 @ w1 + sum2 @ w2 + sum3 @ w3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup-repeat',
                        type=int,
                        default=10,
                        dest='warmup_num')
    parser.add_argument('--timing-repeat',
                        type=int,
                        default=100,
                        dest='test_num')
    cmd_args = parser.parse_args()

    adj = load_txt("../adj.in", "int32")
    n_faces = adj.shape[0]
    in_feats = 13
    out_feats = 64
    x = load_txt("../x.in", "float32")
    w0 = load_txt("../w0.in", "float32")
    w1 = load_txt("../w1.in", "float32")
    w2 = load_txt("../w2.in", "float32")
    w3 = load_txt("../w3.in", "float32")
    d_y = load_txt("../d_y.in", "float32")

    adj = jax.device_put(adj)
    x = jax.device_put(x)
    w0 = jax.device_put(w0)
    w1 = jax.device_put(w1)
    w2 = jax.device_put(w2)
    w3 = jax.device_put(w3)
    d_y = jax.device_put(d_y)

    print(
        f"{cmd_args.warmup_num} warmup, {cmd_args.test_num} repeats for evalution"
    )
    warmup_num = cmd_args.warmup_num
    test_num = cmd_args.test_num

    conv_impl2_inference = jax.jit(conv_impl2)
    # NOTE: JAX requires to compute gradients w.r.t. a scalar, so we sum the output to compute it.
    #       We explicitly multiply d_y here, so it is mathematically equivalent to compute gradients
    #       given d_y
    conv_impl2_forward_backward = jax.grad(
        lambda *args: jnp.sum(conv_impl2(*args) * d_y), argnums=(1, 2, 3, 4, 5))

    for i in range(warmup_num):
        y = conv_impl2_inference(adj, x, w0, w1, w2, w3)
        if i == 0:
            store_txt("y.out", y)
    y = y.block_until_ready()
    t0 = time.time()
    for i in range(test_num):
        y = conv_impl2_inference(adj, x, w0, w1, w2, w3)
    y = y.block_until_ready()
    t1 = time.time()
    assert y.shape == (n_faces, out_feats)
    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        d_x, d_w0, d_w1, d_w2, d_w3 = conv_impl2_forward_backward(
            adj, x, w0, w1, w2, w3)
        if i == 0:
            store_txt("d_x.out", d_x)
            store_txt("d_w0.out", d_w0)
            store_txt("d_w1.out", d_w1)
            store_txt("d_w2.out", d_w2)
            store_txt("d_w3.out", d_w3)
    y = y.block_until_ready()
    t0 = time.time()
    for i in range(test_num):
        d_x, d_w0, d_w1, d_w2, d_w3 = conv_impl2_forward_backward(
            adj, x, w0, w1, w2, w3)
    y = y.block_until_ready()
    t1 = time.time()
    assert d_x.shape == x.shape
    assert d_w0.shape == w0.shape
    assert d_w1.shape == w1.shape
    assert d_w2.shape == w2.shape
    assert d_w3.shape == w3.shape
    print(f"Forward+Backward Time = {(t1 - t0) / test_num * 1000} ms")
