import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

h = 64
w = 64


def load_faces(path: str):
    """
    Load a 3D object and returns the adjacency array of the faces


    Parameters
    ----------
    path: str
        Path to a 3D object file, where a `v <x> <y> <z>` line means there is a vertex at coordinate (x, y, z),
        a `f <i> <j> <k>` line means there is a face among vertices i, j and k. Faces are stored in conter-clockwise
        order


    Returns
    -------
    (np.array, np.array)
        ret[0] is an n*3-shaped numpy array, where n is the number of vertices. array[i] = the coordinate (x, y, z)
        ret[1] is an m*3-shaped numpy array, where m is the number of faces. array[i] = each vertices of the face
    """

    vertices = []
    faces = []
    for line in open(path):
        if line.startswith('v'):
            vertices.append(tuple(map(float, line.split()[1:])))
        if line.startswith('f'):
            faces.append(tuple(map(lambda x: int(x) - 1, line.split()[1:])))
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


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
                                    jnp.linspace(0, 1, w)),
                       axis=-1).reshape(h, w, 2)
    face_verts = jax.vmap(lambda v3: jax.vmap(lambda v: vertices[v, :2])(v3))(
        faces)

    norm = lambda v: jnp.sqrt(v[0] * v[0] + v[1] * v[1])
    cross_product = lambda v1, v2: v1[0] * v2[1] - v1[1] * v2[0]

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

    dist_pixel_to_line = lambda v1, v2, pixel: jnp.abs(
        cross_product(pixel - v1, v2 - v1)) / norm(v2 - v1)
    dist_pixel_to_vert = lambda v, pixel: norm(pixel - v)
    dist_pixel_to_face = lambda v1, v2, v3, pixel: jnp.minimum(
        jnp.minimum(
            jnp.minimum(dist_pixel_to_line(v1, v2, pixel),
                        dist_pixel_to_line(v2, v3, pixel)),
            dist_pixel_to_line(v3, v1, pixel)),
        jnp.minimum(
            jnp.minimum(dist_pixel_to_vert(v1, pixel),
                        dist_pixel_to_vert(v2, pixel)),
            dist_pixel_to_vert(v3, pixel)))
    dist = jax.vmap(lambda face: jax.vmap(lambda row: jax.vmap(
        lambda pixel: dist_pixel_to_face(face[0], face[1], face[2], pixel))
                                          (row))(pixels))(face_verts)
    assert dist.shape == (n_faces, h, w)

    d = jnp.where(is_inside, 1, -1) * dist * dist / sigma
    d = jax.nn.sigmoid(d)
    return d


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <obj-file>")
        print("Please set device in main.sh")
        exit(-1)
    obj_file = sys.argv[1]

    vertices, faces = map(jnp.array, load_faces(obj_file))
    n_verts = vertices.shape[0]
    n_faces = faces.shape[0]

    vertices = jax.device_put(vertices)
    faces = jax.device_put(faces)

    warmup_num = 10
    test_num = 100

    rasterize_inference = jax.jit(rasterize)
    # FIXME: Can we remove the `jnp.sum`?
    rasterize_forward_backward = jax.grad(
        lambda *args: jnp.sum(rasterize(*args)), argnums=(0,))

    for i in range(warmup_num):
        y = rasterize_inference(vertices, faces)
    y = y.block_until_ready()
    t0 = time.time()
    for i in range(test_num):
        y = rasterize_inference(vertices, faces)
    y = y.block_until_ready()
    t1 = time.time()
    assert y.shape == (n_faces, h, w)
    print(f"Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        d_vertices, = rasterize_forward_backward(vertices, faces)
    y = y.block_until_ready()
    t0 = time.time()
    for i in range(test_num):
        d_vertices, = rasterize_forward_backward(vertices, faces)
    y = y.block_until_ready()
    t1 = time.time()
    assert d_vertices.shape == vertices.shape
    print(f"Forward+Backward Time = {(t1 - t0) / test_num * 1000} ms")
