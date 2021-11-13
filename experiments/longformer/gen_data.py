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

    # np.savetxt("q.in", q)
    # np.savetxt("k.in", k)
    # np.savetxt("v.in", v)
    # np.savetxt("d_y.in", d_y)
    q.tofile("q.in", sep=' ', format='%.10f')
    k.tofile("k.in", sep=' ', format='%.10f')
    v.tofile("v.in", sep=' ', format='%.10f')
    d_y.tofile("d_y.in", sep=' ', format='%.10f')