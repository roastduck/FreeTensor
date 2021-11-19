import sys
import itertools
import numpy as np


def load_data(data_name: str):
    '''
    Load data from ../data/{data_name}.config and ../data/{data_name}.graph

    Returns (#vertices, #edges, ptr, idx), where ptr and idx forms a CSR format
    '''

    with open(f"data/{data_name}.config", 'r') as f:
        num_v, num_e = map(int, f.readline().strip().split(' '))

    with open(f"data/{data_name}.graph", 'r') as f:
        ptr = np.array(list(map(int, f.readline().strip().split(" "))))
        idx = np.array(list(map(int, f.readline().strip().split(" "))))

    return num_v, num_e, ptr, idx


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <data_name>")
        exit(-1)
    data_name = sys.argv[1]

    num_v, num_e, ptr, idx = load_data(data_name)

    feat_len = 32
    ptr = ptr.astype("int32")
    idx = idx.astype("int32")
    x = np.random.uniform(size=(num_v, feat_len)).astype("float32")
    w = np.random.uniform(size=(feat_len, feat_len)).astype("float32")
    w_attn_1 = np.random.uniform(size=(feat_len,)).astype("float32")
    w_attn_2 = np.random.uniform(size=(feat_len,)).astype("float32")
    d_y = np.random.uniform(size=(num_v, feat_len)).astype('float32')

    np.savetxt("ptr.in", ptr, fmt="%d")
    np.savetxt("idx.in", idx, fmt="%d")
    np.savetxt("x.in", x)
    np.savetxt("w.in", w)
    np.savetxt("w_attn_1.in", w_attn_1)
    np.savetxt("w_attn_2.in", w_attn_2)
    np.savetxt("d_y.in", d_y)
