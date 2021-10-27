import sys
import time
import itertools
import numpy as np
import ir
from ir.libop import *
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


def compile_all(n_faces, in_feats, out_feats, device):
    mtype = device.main_mem_type()

    @ir.transform
    def inference(adj, x, w0, w1, w2, w3, y):
        ir.declare_var(adj, (n_faces, 3), "int32", "input", mtype)
        ir.declare_var(x, (n_faces, in_feats), "float32", "input", mtype)
        ir.declare_var(w0, (in_feats, out_feats), "float32", "input", mtype)
        ir.declare_var(w1, (in_feats, out_feats), "float32", "input", mtype)
        ir.declare_var(w2, (in_feats, out_feats), "float32", "input", mtype)
        ir.declare_var(w3, (in_feats, out_feats), "float32", "input", mtype)
        ir.declare_var(y, (n_faces, out_feats), "float32", "output", mtype)

        for i in range(n_faces):
            sum1 = zeros((in_feats,), "float32", mtype)()
            sum2 = zeros((in_feats,), "float32", mtype)()
            sum3 = zeros((in_feats,), "float32", mtype)()
            for p in range(3):
                add_to(sum1, x[adj[i, p]])
                add_to(sum2, abs(sub(x[adj[i, p]], x[adj[i, (p + 1) % 3]])))
                add_to(sum3, abs(sub(x[adj[i, p]], x[i])))
            y0 = matmul(x[i], w0)
            y1 = matmul(sum1, w1)
            y2 = matmul(sum2, w2)
            y3 = matmul(sum3, w3)
            assign(y[i], add(add(add(y0, y1), y2), y3))

    forward, backward, requires, privdes, _ = ir.grad(
        inference, set(["x", "w0", "w1", "w2", "w3"]), set(["y"]))

    print("# Inference:")
    print(inference)
    s = ir.Schedule(inference)
    s.auto_schedule(device.target())
    f = ir.lower(s.func(), device.target())
    print(f)
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    inference_exe = ir.Driver(inference, code, device)

    print("# Forward:")
    print(forward)
    s = ir.Schedule(forward)
    s.auto_schedule(device.target())
    f = ir.lower(s.func(), device.target())
    print(f)
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    forward_exe = ir.Driver(forward, code, device)

    print("# Backward:")
    print(backward)
    s = ir.Schedule(backward)
    s.auto_schedule(device.target())
    f = ir.lower(s.func(), device.target())
    print(f)
    code = ir.codegen(f, device.target())
    print(ir.debug.with_line_no(code))
    backward_exe = ir.Driver(backward, code, device)

    def run_inference(adj, x, w0, w1, w2, w3, y):
        inference_exe(adj, x, w0, w1, w2, w3, y)
        inference_exe.sync()

    def run_forward(adj, x, w0, w1, w2, w3, y):
        forward_exe(adj, x, w0, w1, w2, w3, y)
        forward_exe.sync()

    def run_backward(adj, x, w0, w1, w2, w3, y, d_y, d_x, d_w0, d_w1, d_w2,
                     d_w3):
        kvs = {}
        kvs[privdes['y']] = d_y
        kvs[requires['x']] = d_x
        kvs[requires['w0']] = d_w0
        kvs[requires['w1']] = d_w1
        kvs[requires['w2']] = d_w2
        kvs[requires['w3']] = d_w3
        backward_exe(adj, x, w0, w1, w2, w3, y, **kvs)
        backward_exe.sync()

    return run_inference, run_forward, run_backward


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
    d_x = np.zeros(x.shape, dtype='float32')
    d_w0 = np.zeros(w0.shape, dtype='float32')
    d_w1 = np.zeros(w1.shape, dtype='float32')
    d_w2 = np.zeros(w2.shape, dtype='float32')
    d_w3 = np.zeros(w3.shape, dtype='float32')
    d_y = np.random.uniform(size=y.shape).astype('float32')

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
    d_x = ir.Array(d_x, ir_dev)
    d_w0 = ir.Array(d_w0, ir_dev)
    d_w1 = ir.Array(d_w1, ir_dev)
    d_w2 = ir.Array(d_w2, ir_dev)
    d_w3 = ir.Array(d_w3, ir_dev)
    d_y = ir.Array(d_y, ir_dev)

    inference, forward, backward = compile_all(n_faces, in_feats, out_feats,
                                               ir_dev)

    test_num = 1000
    inference(adj, x, w0, w1, w2, w3, y)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        inference(adj, x, w0, w1, w2, w3, y)
    t1 = time.time()

    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")

    test_num = 1000
    forward(adj, x, w0, w1, w2, w3, y)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        forward(adj, x, w0, w1, w2, w3, y)
    t1 = time.time()

    print(f"Forward Time = {(t1 - t0) / test_num * 1000} ms")

    test_num = 1000
    backward(adj, x, w0, w1, w2, w3, y, d_y, d_x, d_w0, d_w1, d_w2,
             d_w3)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        backward(adj, x, w0, w1, w2, w3, y, d_y, d_x, d_w0, d_w1, d_w2, d_w3)
    t1 = time.time()

    print(f"Backward Time = {(t1 - t0) / test_num * 1000} ms")
