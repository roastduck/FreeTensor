import sys
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp

sys.path.append('../..')
from common.jax.io import load_txt, store_txt

h = 64
w = 64


def rasterize(vertices, faces):
    """
    Compute soft rasterization of each faces

    Suppose the points are already transposed, so we are viewing inside 0 <= x <= 1 and 0 <= y <= 1, along z-axis.
    The resolution along x and y is h and w, correspondingly.

    Returns
    -------
    jnp.array
        An h*w*m-shaped tensor, where m is the number of faces, tensor[i, j, k] = the probability of face k at
        pixel (i, j)
    """

    n_verts = vertices.shape[0]
    n_faces = faces.shape[0]
    assert vertices.shape == (n_verts, 3)
    assert faces.shape == (n_faces, 3)

    sigma = 1e-4

    pixels = jnp.stack(jnp.meshgrid(jnp.linspace(0, 1, h),
                                    jnp.linspace(0, 1, w),
                                    indexing='ij'),
                       axis=-1).reshape(h, w, 2)
    face_verts = jax.vmap(lambda v3: jax.vmap(lambda v: vertices[v, :2])(v3))(
        faces)

    norm = lambda v: jnp.sqrt(v[0] * v[0] + v[1] * v[1])
    cross_product = lambda v1, v2: v1[0] * v2[1] - v1[1] * v2[0]
    dot_product = lambda v1, v2: v1[0] * v2[0] + v1[1] * v2[1]

    vert_clockwise = lambda v1, v2, pixel: cross_product(pixel - v1, v2 - v1
                                                        ) < 0
    inside_face = lambda v1, v2, v3, pixel: jnp.logical_and(
        jnp.logical_and(vert_clockwise(v1, v2, pixel),
                        vert_clockwise(v2, v3, pixel)),
        vert_clockwise(v3, v1, pixel))
    is_inside = jax.vmap(
        lambda face: jax.vmap(lambda row: jax.vmap(lambda pixel: inside_face(
            face[0], face[1], face[2], pixel))(row))(pixels))(face_verts)
    assert is_inside.shape == (n_faces, h, w)

    dist_pixel_to_seg = lambda v1, v2, pixel: jnp.where(
        dot_product(pixel - v1, v2 - v1) >= 0,
        jnp.where(
            dot_product(pixel - v2, v1 - v2) >= 0,
            jnp.abs(cross_product(pixel - v1, v2 - v1)) / norm(v2 - v1),
            norm(pixel - v2)), norm(pixel - v1))
    dist_pixel_to_face = lambda v1, v2, v3, pixel: jnp.minimum(
        jnp.minimum(dist_pixel_to_seg(v1, v2, pixel),
                    dist_pixel_to_seg(v2, v3, pixel)),
        dist_pixel_to_seg(v3, v1, pixel))
    dist = jax.vmap(lambda face: jax.vmap(lambda row: jax.vmap(
        lambda pixel: dist_pixel_to_face(face[0], face[1], face[2], pixel))
                                          (row))(pixels))(face_verts)
    assert dist.shape == (n_faces, h, w)

    d = jnp.where(is_inside, 1, -1) * dist * dist / sigma
    d = jax.nn.sigmoid(d)
    return d


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
    parser.add_argument('--profile-gpu',
                        action='store_true',
                        dest='profile_gpu')
    cmd_args = parser.parse_args()

    if cmd_args.profile_gpu:
        from common.gpu import profile_start, profile_stop

    vertices = load_txt("../vertices.in", "float32")
    faces = load_txt("../faces.in", "int32")
    d_y = load_txt("../d_y.in", "float32")
    n_verts = vertices.shape[0]
    n_faces = faces.shape[0]

    vertices = jax.device_put(vertices)
    faces = jax.device_put(faces)
    d_y = jax.device_put(d_y)

    print(
        f"{cmd_args.warmup_num} warmup, {cmd_args.test_num} repeats for evalution"
    )
    warmup_num = cmd_args.warmup_num
    test_num = cmd_args.test_num

    rasterize_inference = jax.jit(rasterize)
    # NOTE: JAX requires to compute gradients w.r.t. a scalar, so we sum the output to compute it.
    #       We explicitly multiply d_y here, so it is mathematically equivalent to compute gradients
    #       given d_y
    rasterize_forward_backward = jax.grad(
        lambda *args: jnp.sum(rasterize(*args) * d_y), argnums=(0,))

    for i in range(warmup_num):
        y = rasterize_inference(vertices, faces)
        if i == 0:
            store_txt("y.out", y)
    y = y.block_until_ready()
    if cmd_args.profile_gpu:
        profile_start()
    t0 = time.time()
    for i in range(test_num):
        y = rasterize_inference(vertices, faces)
    y = y.block_until_ready()
    t1 = time.time()
    if cmd_args.profile_gpu:
        profile_stop()
    assert y.shape == (n_faces, h, w)
    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")

    if cmd_args.profile_gpu:
        exit(0)

    for i in range(warmup_num):
        d_vertices, = rasterize_forward_backward(vertices, faces)
        if i == 0:
            store_txt("d_vertices.out", d_vertices)
    y = y.block_until_ready()
    t0 = time.time()
    for i in range(test_num):
        d_vertices, = rasterize_forward_backward(vertices, faces)
    y = y.block_until_ready()
    t1 = time.time()
    assert d_vertices.shape == vertices.shape
    print(f"Forward+Backward Time = {(t1 - t0) / test_num * 1000} ms")
