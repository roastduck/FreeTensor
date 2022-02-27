import sys
import numpy as np

sys.path.append('..')
from common.numpy.io import store_txt

if __name__ == '__main__':
    n_heads = 8
    seq_len = 10000
    feat_len = 512
    w = 32
    dilation = 4  # counts from 1
    dilation_heads = 2

    q = np.random.uniform(size=(n_heads, seq_len, feat_len)).astype("float32")
    k = np.random.uniform(size=(n_heads, seq_len, feat_len)).astype("float32")
    v = np.random.uniform(size=(n_heads, seq_len, feat_len)).astype("float32")
    d_y = np.random.uniform(size=(n_heads, seq_len, feat_len)).astype('float32')

    store_txt("q.in", q)
    store_txt("k.in", k)
    store_txt("v.in", v)
    store_txt("d_y.in", d_y)
