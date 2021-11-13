import numpy as np
if __name__ == '__main__':

    length = 100
    in_feats = 4
    hidden_feats = 256

    x = np.random.uniform(size=(length, in_feats)).astype("float32")
    wf = np.random.uniform(size=(hidden_feats, in_feats)).astype("float32")
    wi = np.random.uniform(size=(hidden_feats, in_feats)).astype("float32")
    wo = np.random.uniform(size=(hidden_feats, in_feats)).astype("float32")
    wc = np.random.uniform(size=(hidden_feats, in_feats)).astype("float32")
    uf = np.random.uniform(size=(hidden_feats, hidden_feats)).astype("float32")
    ui = np.random.uniform(size=(hidden_feats, hidden_feats)).astype("float32")
    uo = np.random.uniform(size=(hidden_feats, hidden_feats)).astype("float32")
    uc = np.random.uniform(size=(hidden_feats, hidden_feats)).astype("float32")
    bf = np.random.uniform(size=(hidden_feats, )).astype("float32")
    bi = np.random.uniform(size=(hidden_feats, )).astype("float32")
    bo = np.random.uniform(size=(hidden_feats, )).astype("float32")
    bc = np.random.uniform(size=(hidden_feats, )).astype("float32")
    d_y = np.random.uniform(size=(hidden_feats, )).astype('float32')

    np.savetxt("x.in", x)
    np.savetxt("wf.in", wf)
    np.savetxt("wi.in", wi)
    np.savetxt("wo.in", wo)
    np.savetxt("wc.in", wc)
    np.savetxt("uf.in", uf)
    np.savetxt("ui.in", ui)
    np.savetxt("uo.in", uo)
    np.savetxt("uc.in", uc)
    np.savetxt("bf.in", bf)
    np.savetxt("bi.in", bi)
    np.savetxt("bo.in", bo)
    np.savetxt("bc.in", bc)
    np.savetxt("d_y.in", d_y)
