import sys
import time
import math
import numpy as np
import jax
import jax.numpy as jnp

n_heads = 8
seq_len = 10000
feat_len = 512
w = 32
dilation = 4  # counts from 1
dilation_heads = 2


def dilated_attention(q, k, v, dilation):
    n_heads, seq_len, feat_len = q.shape
    assert q.shape == (n_heads, seq_len, feat_len)
    assert k.shape == (n_heads, seq_len, feat_len)
    assert v.shape == (n_heads, seq_len, feat_len)

    sqrt_d = math.sqrt(feat_len)

    # NOTE 1: JAX does not support strides so we cannot implement with as_strided
    # NOTE 2: JAX requires static array slicing so we cannot implement with things like vmap(head[i:i+2w+1])
    #         Error message: jax._src.traceback_util.UnfilteredStackTrace: IndexError: Array slice indices must
    #                        have static start/stop/step to be used with NumPy indexing syntax. To index a statically
    #                        sized array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice (JAX does
    #                        not support dynamically sized arrays within JIT compiled functions).
    # NOTE 3: JAX does not support slice assignments so we cannot create an empty array and then fill it
    #         Error message: jax._src.traceback_util.UnfilteredStackTrace: TypeError: JAX 'Tracer' objects do not
    #                        support item assignment
    # Tested with JAX 0.2.19

    pad_k = jnp.pad(k, ((0, 0), (w * dilation, w * dilation), (0, 0)))
    pad_v = jnp.pad(v, ((0, 0), (w * dilation, w * dilation), (0, 0)))
    assert pad_k.shape == (n_heads, seq_len + 2 * w * dilation, feat_len)
    assert pad_v.shape == (n_heads, seq_len + 2 * w * dilation, feat_len)
    diag_k = jax.vmap(lambda head: jax.vmap(lambda i: jax.vmap(lambda j: head[
        i + j * dilation])(jnp.arange(0, 2 * w + 1)))(jnp.arange(0, seq_len)))(
            pad_k)
    diag_v = jax.vmap(lambda head: jax.vmap(lambda i: jax.vmap(lambda j: head[
        i + j * dilation])(jnp.arange(0, 2 * w + 1)))(jnp.arange(0, seq_len)))(
            pad_v)

    attn = jnp.einsum("ijp,ijkp->ijk", q, diag_k)
    assert attn.shape == (n_heads, seq_len, 2 * w + 1)
    attn = jax.nn.softmax(attn, axis=-1) / sqrt_d

    return jnp.einsum("ijk,ijkp->ijp", attn, diag_v)


@jax.jit
def transformer_impl1(q, k, v):
    front_heads = dilated_attention(q[:dilation_heads], k[:dilation_heads],
                                    v[:dilation_heads], dilation)
    back_heads = dilated_attention(q[dilation_heads:], k[dilation_heads:],
                                   v[dilation_heads:], 1)
    return jnp.concatenate([front_heads, back_heads], axis=0)


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    q = jax.random.uniform(key, (n_heads, seq_len, feat_len), dtype=jnp.float32)
    k = jax.random.uniform(key, (n_heads, seq_len, feat_len), dtype=jnp.float32)
    v = jax.random.uniform(key, (n_heads, seq_len, feat_len), dtype=jnp.float32)

    q = jax.device_put(q)
    k = jax.device_put(k)
    v = jax.device_put(v)

    test_num = 100

    y = transformer_impl1(q, k, v)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        y = transformer_impl1(q, k, v)
    t1 = time.time()
    assert y.shape == (n_heads, seq_len, feat_len)
    print(f"Time = {(t1 - t0) / test_num * 1000} ms")
