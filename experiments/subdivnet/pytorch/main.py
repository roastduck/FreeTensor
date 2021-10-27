import sys
import time
import itertools
import numpy as np
import torch


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

    adj_feat = torch.index_select(x, 0,
                                  adj.flatten()).reshape(n_faces, 3, in_feats)
    y0 = x @ w0
    y1 = torch.sum(adj_feat, dim=1) @ w1
    y2 = torch.sum(
        torch.abs(adj_feat -
                  torch.cat([adj_feat[:, 1:], adj_feat[:, :1]], dim=1)),
        dim=1) @ w2
    y3 = torch.sum(torch.abs(adj_feat - x.reshape(n_faces, 1, in_feats)),
                   dim=1) @ w3
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

    adj_feat = torch.index_select(x, 0,
                                  adj.flatten()).reshape(n_faces, 3, in_feats)
    sum1 = torch.zeros_like(x)
    sum2 = torch.zeros_like(x)
    sum3 = torch.zeros_like(x)
    for p in range(3):
        sum1 += adj_feat[:, p]
        sum2 += torch.abs(adj_feat[:, p] - adj_feat[:, (p + 1) % 3])
        sum3 += torch.abs(adj_feat[:, p] - x)

    return x @ w0 + sum1 @ w1 + sum2 @ w2 + sum3 @ w3


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <cpu/gpu> <obj-file>")
        exit(-1)
    device = sys.argv[1]
    obj_file = sys.argv[2]

    adj = torch.tensor(load_faces(obj_file))
    n_faces = adj.shape[0]
    in_feats = 13
    out_feats = 64
    x = torch.rand(n_faces, in_feats, dtype=torch.float)
    w0 = torch.rand(in_feats, out_feats, dtype=torch.float)
    w1 = torch.rand(in_feats, out_feats, dtype=torch.float)
    w2 = torch.rand(in_feats, out_feats, dtype=torch.float)
    w3 = torch.rand(in_feats, out_feats, dtype=torch.float)
    d_y = torch.rand(n_faces, out_feats, dtype=torch.float)

    if device == 'gpu':
        adj = adj.cuda()
        x = x.cuda()
        w0 = w0.cuda()
        w1 = w1.cuda()
        w2 = w2.cuda()
        w3 = w3.cuda()
        d_y = d_y.cuda()
    else:
        assert device == 'cpu'

    test_num = 1000

    y = conv_impl1(adj, x, w0, w1, w2, w3)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        y = conv_impl1(adj, x, w0, w1, w2, w3)
    t1 = time.time()
    assert y.shape == (n_faces, out_feats)
    print(f"Impl1 Inference Time = {(t1 - t0) / test_num * 1000} ms")

    y = conv_impl2(adj, x, w0, w1, w2, w3)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        y = conv_impl2(adj, x, w0, w1, w2, w3)
    t1 = time.time()
    assert y.shape == (n_faces, out_feats)
    print(f"Impl2 Inference Time = {(t1 - t0) / test_num * 1000} ms")

    x.requires_grad = True
    w0.requires_grad = True
    w1.requires_grad = True
    w2.requires_grad = True
    w3.requires_grad = True

    y = conv_impl1(adj, x, w0, w1, w2, w3)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        y = conv_impl1(adj, x, w0, w1, w2, w3)
    t1 = time.time()
    assert y.shape == (n_faces, out_feats)
    print(f"Impl1 Forward Time = {(t1 - t0) / test_num * 1000} ms")

    y.backward(d_y, retain_graph=True)
    t0 = time.time()
    for i in range(test_num):
        y.backward(d_y, retain_graph=True)
    t1 = time.time()
    print(f"Impl1 Backward Time = {(t1 - t0) / test_num * 1000} ms")

    y = conv_impl2(adj, x, w0, w1, w2, w3)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        y = conv_impl2(adj, x, w0, w1, w2, w3)
    t1 = time.time()
    assert y.shape == (n_faces, out_feats)
    print(f"Impl2 Forward Time = {(t1 - t0) / test_num * 1000} ms")

    y.backward(d_y, retain_graph=True)
    t0 = time.time()
    for i in range(test_num):
        y.backward(d_y, retain_graph=True)
    t1 = time.time()
    print(f"Impl2 Backward Time = {(t1 - t0) / test_num * 1000} ms")
