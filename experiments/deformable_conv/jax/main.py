import sys
import time
import math
import numpy as np
import jax
import jax.numpy as jnp


@jax.jit
def conv_impl1(x, w1, w2):
    n, c_in, h, w = x.shape
    c_out = w2.shape[0]
    k_h = w1.shape[0]
    k_w = w1.shape[1]
    assert x.shape == (n, c_in, h, w)
    assert w1.shape == (k_h, k_w, 2, c_in, k_h, k_w)
    assert w2.shape == (c_out, c_in, k_h, k_w)

    offset = jax.lax.conv(x, w1.reshape(-1, c_in, k_h, k_w), [1, 1],
                          "SAME").reshape(n, k_h, k_w, 2, h, w)
    offset /= c_in  # Make offset not too large
    offset = offset.transpose(0, 4, 5, 1, 2, 3)
    assert offset.shape == (n, h, w, k_h, k_w, 2)

    offset += jnp.stack(jnp.meshgrid(jnp.arange(-(k_h // 2), k_h - k_h // 2),
                                     jnp.arange(-(k_w // 2), k_w - k_w // 2),
                                     indexing='ij'),
                        axis=-1).reshape(k_h, k_w, 2)

    coords = jnp.stack(jnp.meshgrid(jnp.arange(h), jnp.arange(w),
                                    indexing='ij'),
                       axis=-1).reshape(h, w, 1, 1, 2)
    coords = coords + offset
    assert coords.shape == (n, h, w, k_h, k_w, 2)

    coords_int = jnp.floor(coords).astype(jnp.int32)
    # floor is necessary because floor(-1.5) = -2

    # We apply an 1-pixel wide padding, and limit the coordinate to not exceeding the padding
    # Rational:
    # 1. jnp.select computes both cases and then select, which is infeasible because of out-of-range array accesses
    # 2. Python `if` in a `vmap` is not supported
    # 3. A dynamic-sized padding is not suppoted by JAX
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    img_or_pad = lambda img_pad, row, col: img_pad[jnp.clip(row + 1, 0, h + 1),
                                                   jnp.clip(col + 1, 0, w + 1)]

    def mix_pixel(img_pad, row, col, row_fp, col_fp):
        ret = img_or_pad(img_pad, row, col) * (row_fp - row) * (col_fp - col)
        ret += img_or_pad(img_pad, row,
                          col + 1) * (row_fp - row) * (col + 1 - col_fp)
        ret += img_or_pad(img_pad, row + 1,
                          col) * (row + 1 - row_fp) * (col_fp - col)
        ret += img_or_pad(img_pad, row + 1,
                          col + 1) * (row + 1 - row_fp) * (col + 1 - col_fp)
        return ret

    pixels = jax.vmap(lambda sample, sample_c, sample_c_fp: jax.vmap(
        lambda img: jax.vmap(lambda c, c_fp: mix_pixel(img, c[0], c[1], c_fp[
            0], c_fp[1]))(sample_c, sample_c_fp))(sample))(
                x_pad, coords_int.reshape(n, -1, 2),
                coords.reshape(n, -1, 2)).reshape(n, c_in, h, w, k_h, k_w)

    return jnp.einsum("nchwrs,kcrs->nkhw", pixels, w2)


if __name__ == '__main__':
    n = 8
    c_in = 256
    c_out = 256
    h = 56
    w = 56
    k_h = 3
    k_w = 3

    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (n, c_in, h, w), dtype=jnp.float32) * 2 - 1
    w1 = jax.random.uniform(key, (k_h, k_w, 2, c_in, k_h, k_w),
                            dtype=jnp.float32) * 2 - 1
    w2 = jax.random.uniform(key,
                            (c_out, c_in, k_h, k_w), dtype=jnp.float32) * 2 - 1

    x = jax.device_put(x)
    w1 = jax.device_put(w1)
    w2 = jax.device_put(w2)

    test_num = 100

    y = conv_impl1(x, w1, w2)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        y = conv_impl1(x, w1, w2)
    t1 = time.time()
    assert y.shape == (n, c_out, h, w)
    print(f"Time = {(t1 - t0) / test_num * 1000} ms")
