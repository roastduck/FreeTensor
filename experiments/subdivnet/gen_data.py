import sys
import itertools
import numpy as np

sys.path.append('..')
from common.numpy.io import store_txt


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


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <obj-file>")
        exit(-1)
    obj_file = sys.argv[1]

    adj = load_faces(obj_file)
    n_faces = adj.shape[0]
    in_feats = 13
    out_feats = 64

    x = np.random.uniform(size=(n_faces, in_feats)).astype("float32")
    w0 = np.random.uniform(size=(in_feats, out_feats)).astype("float32")
    w1 = np.random.uniform(size=(in_feats, out_feats)).astype("float32")
    w2 = np.random.uniform(size=(in_feats, out_feats)).astype("float32")
    w3 = np.random.uniform(size=(in_feats, out_feats)).astype("float32")
    d_y = np.random.uniform(size=(n_faces, out_feats)).astype('float32')

    store_txt("adj.in", adj)
    store_txt("x.in", x)
    store_txt("w0.in", w0)
    store_txt("w1.in", w1)
    store_txt("w2.in", w2)
    store_txt("w3.in", w3)
    store_txt("d_y.in", d_y)
