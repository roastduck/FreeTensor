import sys
import time
import numpy as np
import torch


def rasterize(vertices, faces, h, w):
    """
    Compute soft rasterization of each faces

    Suppose the points are already transposed, so we are viewing inside 0 <= x <= 1 and 0 <= y <= 1, along z-axis.
    The resolution along x and y is h and w, correspondingly.

    Returns
    -------
    torch.Tensor
        An h*w*m-shaped tensor, where m is the number of faces, tensor[i, j, k] = the probability of face k at
        pixel (i, j)
    """

    n_verts = vertices.shape[0]
    n_faces = faces.shape[0]
    assert vertices.shape == (n_verts, 3)
    assert faces.shape == (n_faces, 3)

    sigma = 1e-4

    pixels = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, h, device=faces.device),
        torch.linspace(0, 1, w, device=faces.device)),
                         dim=-1).reshape(h, w, 2)
    face_verts = torch.index_select(vertices, 0,
                                    faces.flatten()).reshape(n_faces, 3,
                                                             3)[:, :, :2]

    norm = lambda v: torch.sqrt(
        v.select(-1, 0) * v.select(-1, 0) + v.select(-1, 1) * v.select(-1, 1))
    cross_product = lambda v1, v2: v1.select(-1, 0) * v2.select(
        -1, 1) - v1.select(-1, 1) * v2.select(-1, 0)
    dot_product = lambda v1, v2: v1.select(-1, 0) * v2.select(
        -1, 0) + v1.select(-1, 1) * v2.select(-1, 1)

    vert_clockwise = lambda v1, v2, pixel: cross_product(pixel - v1, v2 - v1
                                                        ) < 0
    inside_face = lambda v1, v2, v3, pixel: torch.logical_and(
        torch.logical_and(vert_clockwise(v1, v2, pixel),
                          vert_clockwise(v2, v3, pixel)),
        vert_clockwise(v3, v1, pixel))
    is_inside = inside_face(face_verts[:, 0, :].reshape(n_faces, 1, 1, 2),
                            face_verts[:, 1, :].reshape(n_faces, 1, 1, 2),
                            face_verts[:, 2, :].reshape(n_faces, 1, 1, 2),
                            pixels.reshape(1, h, w, 2))
    assert is_inside.shape == (n_faces, h, w)

    dist_pixel_to_seg = lambda v1, v2, pixel: torch.where(
        dot_product(pixel - v1, v2 - v1) >= 0,
        torch.where(
            dot_product(pixel - v2, v1 - v2) >= 0,
            torch.abs(cross_product(pixel - v1, v2 - v1)) / norm(v2 - v1),
            norm(pixel - v2)), norm(pixel - v1))

    dist_pixel_to_face = lambda v1, v2, v3, pixel: torch.minimum(
        torch.minimum(dist_pixel_to_seg(v1, v2, pixel),
                      dist_pixel_to_seg(v2, v3, pixel)),
        dist_pixel_to_seg(v3, v1, pixel))
    dist = dist_pixel_to_face(face_verts[:, 0, :].reshape(n_faces, 1, 1, 2),
                              face_verts[:, 1, :].reshape(n_faces, 1, 1, 2),
                              face_verts[:, 2, :].reshape(n_faces, 1, 1, 2),
                              pixels.reshape(1, h, w, 2))
    assert dist.shape == (n_faces, h, w)

    d = torch.where(is_inside, 1, -1) * dist * dist / sigma
    d = torch.sigmoid(d)
    return d


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu/gpu>")
        exit(-1)
    device = sys.argv[1]

    vertices = torch.tensor(np.load("../vertices.in.npy"), dtype=torch.float)
    faces = torch.tensor(np.load("../faces.in.npy"))
    n_verts = vertices.shape[0]
    n_faces = faces.shape[0]
    h = 64
    w = 64
    d_y = torch.tensor(np.load("../d_y.in.npy"), dtype=torch.float)

    if device == 'gpu':
        vertices = vertices.cuda()
        faces = faces.cuda()
        d_y = d_y.cuda()
        sync = torch.cuda.synchronize
    else:
        assert device == 'cpu'
        sync = lambda: None

    warmup_num = 10
    test_num = 100

    for i in range(warmup_num):
        y = rasterize(vertices, faces, h, w)
        if i == 0:
            np.save("y.out.npy", y.cpu().numpy(), allow_pickle=False)
    sync()
    t0 = time.time()
    for i in range(test_num):
        y = rasterize(vertices, faces, h, w)
    sync()
    t1 = time.time()
    assert y.shape == (n_faces, h, w)
    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")

    vertices.requires_grad = True

    for i in range(warmup_num):
        y = rasterize(vertices, faces, h, w)
    sync()
    t0 = time.time()
    for i in range(test_num):
        y = rasterize(vertices, faces, h, w)
    sync()
    t1 = time.time()
    assert y.shape == (n_faces, h, w)
    print(f"Forward Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        y.backward(d_y, retain_graph=True)
        if i == 0:
            np.save("d_vertices.out.npy",
                    vertices.grad.cpu().numpy(),
                    allow_pickle=False)
    sync()
    t0 = time.time()
    for i in range(test_num):
        y.backward(d_y, retain_graph=True)
    sync()
    t1 = time.time()
    print(f"Backward Time = {(t1 - t0) / test_num * 1000} ms")
