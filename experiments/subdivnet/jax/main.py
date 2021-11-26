import sys
import time
import itertools
import numpy as np
import jax
import jax.numpy as jnp


def load_faces(path: str):
    """
    Load a 3D object and returns the adjacency array of the faces


    Parameters
    ----------
    path: str
        Path to a 3D object file, where a `f <i> <j> <k>` line means there is a face among point i, j and k


    Returns
    -------
    np.array
        An n*3-shaped numpy array, where n is the number of faces. array[i][j] = ID of the j-th adjacent face of the i-th face
    """

    faces = []
    for line in open(path):
        if line.startswith('f'):
            faces.append(tuple(map(int, line.split()[1:])))

    edgeToFaces = {}
    for face, i in zip(faces, itertools.count()):
        edgeToFaces[(face[0], face[1])] = i
        edgeToFaces[(face[1], face[2])] = i
        edgeToFaces[(face[2], face[0])] = i

    ret = []
    for face, i in zip(faces, itertools.count()):
        ret.append(
            (edgeToFaces[(face[1], face[0])], edgeToFaces[(face[2], face[1])],
             edgeToFaces[(face[0], face[2])]))

    return np.array(ret, dtype=np.int32)


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

    adj_feat = jax.vmap(lambda f3: jax.vmap(lambda f: x[f])(f3))(adj)
    y0 = x @ w0
    y1 = jnp.sum(adj_feat, axis=1) @ w1
    y2 = jnp.sum(
        jnp.abs(adj_feat -
                jnp.concatenate([adj_feat[:, 1:], adj_feat[:, :1]], axis=1)),
        axis=1) @ w2
    y3 = jnp.sum(jnp.abs(adj_feat - x.reshape(n_faces, 1, in_feats)),
                 axis=1) @ w3
    return y0 + y1 + y2 + y3


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
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <obj-file>")
        print("Please set device in main.sh")
        exit(-1)
    obj_file = sys.argv[1]

    adj = jnp.array(load_faces(obj_file))
    n_faces = adj.shape[0]
    in_feats = 13
    out_feats = 64
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (n_faces, in_feats), dtype=jnp.float32)
    w0 = jax.random.uniform(key, (in_feats, out_feats), dtype=jnp.float32)
    w1 = jax.random.uniform(key, (in_feats, out_feats), dtype=jnp.float32)
    w2 = jax.random.uniform(key, (in_feats, out_feats), dtype=jnp.float32)
    w3 = jax.random.uniform(key, (in_feats, out_feats), dtype=jnp.float32)

    adj = jax.device_put(adj)
    x = jax.device_put(x)
    w0 = jax.device_put(w0)
    w1 = jax.device_put(w1)
    w2 = jax.device_put(w2)
    w3 = jax.device_put(w3)

    warmup_num = 10
    test_num = 1000

    conv_impl1_inference = jax.jit(conv_impl1)
    conv_impl2_inference = jax.jit(conv_impl2)
    # FIXME: Can we remove the `jnp.sum`?
    conv_impl1_forward_backward = jax.grad(
        lambda *args: jnp.sum(conv_impl1(*args)), argnums=(1, 2, 3, 4, 5))
    conv_impl2_forward_backward = jax.grad(
        lambda *args: jnp.sum(conv_impl2(*args)), argnums=(1, 2, 3, 4, 5))

    for i in range(warmup_num):
        y = conv_impl1_inference(adj, x, w0, w1, w2, w3)
    y = y.block_until_ready()
    t0 = time.time()
    for i in range(test_num):
        y = conv_impl1_inference(adj, x, w0, w1, w2, w3)
    y = y.block_until_ready()
    t1 = time.time()
    assert y.shape == (n_faces, out_feats)
    print(f"Impl1 Inference Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        y = conv_impl2_inference(adj, x, w0, w1, w2, w3)
    y = y.block_until_ready()
    t0 = time.time()
    for i in range(test_num):
        y = conv_impl2_inference(adj, x, w0, w1, w2, w3)
    y = y.block_until_ready()
    t1 = time.time()
    assert y.shape == (n_faces, out_feats)
    print(f"Impl2 Inference Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        d_x, d_w0, d_w1, d_w2, d_w3 = conv_impl1_forward_backward(
            adj, x, w0, w1, w2, w3)
    y = y.block_until_ready()
    t0 = time.time()
    for i in range(test_num):
        d_x, d_w0, d_w1, d_w2, d_w3 = conv_impl1_forward_backward(
            adj, x, w0, w1, w2, w3)
    y = y.block_until_ready()
    t1 = time.time()
    assert d_x.shape == x.shape
    assert d_w0.shape == w0.shape
    assert d_w1.shape == w1.shape
    assert d_w2.shape == w2.shape
    assert d_w3.shape == w3.shape
    print(f"Impl1 Forward+Backward Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        d_x, d_w0, d_w1, d_w2, d_w3 = conv_impl2_forward_backward(
            adj, x, w0, w1, w2, w3)
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
    print(f"Impl2 Forward+Backward Time = {(t1 - t0) / test_num * 1000} ms")
