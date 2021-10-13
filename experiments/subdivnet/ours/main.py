import sys
import time
import itertools
import numpy as np
import ir
import ir.libop as lop
import ir.debug


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


jit_cache = {}


def conv(adj, x, w0, w1, w2, w3, y, n_faces, in_feats, out_feats, device):
    # TODO: Dilation
    # TODO: Stride
    # TODO: Batch

    if (n_faces, in_feats, out_feats) in jit_cache:
        exe = jit_cache[(n_faces, in_feats, out_feats)]

    else:
        mtype = device.main_mem_type()

        @ir.transform
        def f(adj, x, w0, w1, w2, w3, y):
            ir.declare_var(adj, (n_faces, 3), "int32", "input", mtype)
            ir.declare_var(x, (n_faces, in_feats), "float32", "input", mtype)
            ir.declare_var(w0, (in_feats, out_feats), "float32", "input", mtype)
            ir.declare_var(w1, (in_feats, out_feats), "float32", "input", mtype)
            ir.declare_var(w2, (in_feats, out_feats), "float32", "input", mtype)
            ir.declare_var(w3, (in_feats, out_feats), "float32", "input", mtype)
            ir.declare_var(y, (n_faces, out_feats), "float32", "output", mtype)

            for i in range(n_faces):
                sum1 = lop.zeros((in_feats,), "float32", mtype)()
                sum2 = lop.zeros((in_feats,), "float32", mtype)()
                sum3 = lop.zeros((in_feats,), "float32", mtype)()
                for p in range(3):
                    lop.add_to(sum1, x[adj[i, p]])
                    lop.add_to(
                        sum2,
                        lop.abs(lop.sub(x[adj[i, p]], x[adj[i, (p + 1) % 3]])))
                    lop.add_to(sum3, lop.abs(lop.sub(x[adj[i, p]], x[i])))
                for j in range(out_feats):
                    y[i, j] = 0.
                    for k in range(in_feats):
                        y[i, j] += x[i, k] * w0[k, j] + sum1[k] * w1[
                            k, j] + sum2[k] * w2[k, j] + sum3[k] * w3[k, j]

        s = ir.Schedule(f)
        s.auto_schedule(device.target())
        f = ir.lower(s.func(), device.target())
        print(f)
        code = ir.codegen(f, device.target())
        print(ir.debug.with_line_no(code))
        exe = ir.Driver(f, code, device)
        exe.set_params(adj, x, w0, w1, w2, w3, y)
        # TODO: do not set_params here
        jit_cache[(n_faces, in_feats, out_feats)] = exe

    exe.run()
    exe.sync()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <cpu/gpu> <obj-file>")
        exit(-1)
    device = sys.argv[1]
    obj_file = sys.argv[2]

    adj = load_faces(obj_file)
    n_faces = adj.shape[0]
    in_feats = 13
    out_feats = 64
    x = np.random.uniform(size=(n_faces, in_feats)).astype("float32")
    w0 = np.random.uniform(size=(in_feats, out_feats)).astype("float32")
    w1 = np.random.uniform(size=(in_feats, out_feats)).astype("float32")
    w2 = np.random.uniform(size=(in_feats, out_feats)).astype("float32")
    w3 = np.random.uniform(size=(in_feats, out_feats)).astype("float32")
    y = np.zeros((n_faces, out_feats), dtype="float32")

    if device == 'gpu':
        ir_dev = ir.Device(ir.GPU())
    else:
        assert device == 'cpu'
        ir_dev = ir.Device(ir.CPU())

    adj = ir.Array(adj, ir_dev)
    x = ir.Array(x, ir_dev)
    w0 = ir.Array(w0, ir_dev)
    w1 = ir.Array(w1, ir_dev)
    w2 = ir.Array(w2, ir_dev)
    w3 = ir.Array(w3, ir_dev)
    y = ir.Array(y, ir_dev)

    test_num = 1000
    conv(adj, x, w0, w1, w2, w3, y, n_faces, in_feats, out_feats,
         ir_dev)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        conv(adj, x, w0, w1, w2, w3, y, n_faces, in_feats, out_feats, ir_dev)
    t1 = time.time()

    print(f"Time = {(t1 - t0) / test_num * 1000} ms")
