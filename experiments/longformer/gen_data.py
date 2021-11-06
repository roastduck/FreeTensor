import numpy as np

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

    np.save("q.in.npy", q, allow_pickle=False)
    np.save("k.in.npy", k, allow_pickle=False)
    np.save("v.in.npy", v, allow_pickle=False)
    np.save("d_y.in.npy", d_y, allow_pickle=False)
