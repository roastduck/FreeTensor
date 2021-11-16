import numpy as np
if __name__ == '__main__':

    length = 100
    in_feats = 4
    hidden_feats = 256
    div = 300

    x = np.random.uniform(size=(length, in_feats)).astype("float32")
    wf = np.random.uniform(size=(hidden_feats, in_feats)).astype("float32") / div
    wi = np.random.uniform(size=(hidden_feats, in_feats)).astype("float32") / div
    wo = np.random.uniform(size=(hidden_feats, in_feats)).astype("float32") / div
    wc = np.random.uniform(size=(hidden_feats, in_feats)).astype("float32") / div
    uf = np.random.uniform(size=(hidden_feats, hidden_feats)).astype("float32") / div
    ui = np.random.uniform(size=(hidden_feats, hidden_feats)).astype("float32") / div
    uo = np.random.uniform(size=(hidden_feats, hidden_feats)).astype("float32") / div
    uc = np.random.uniform(size=(hidden_feats, hidden_feats)).astype("float32") / div
    bf = np.random.uniform(size=(hidden_feats, )).astype("float32") / div
    bi = np.random.uniform(size=(hidden_feats, )).astype("float32") / div
    bo = np.random.uniform(size=(hidden_feats, )).astype("float32") / div
    bc = np.random.uniform(size=(hidden_feats, )).astype("float32") / div
    d_y = np.random.uniform(size=(hidden_feats, )).astype('float32') / div

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
