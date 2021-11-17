import numpy as np
import torch

if __name__ == '__main__':

    length = 100
    in_feats = 4
    hidden_feats = 256

    # NOTE: LSTM requires special initialization for numerical stability


    def init_weight(shape):
        w = torch.empty(shape, dtype=torch.float)
        torch.nn.init.xavier_uniform_(w)
        return w.numpy()

    def init_norm(shape):
        return torch.randn(shape, dtype=torch.float).numpy()

    x = init_norm((length, in_feats))
    wf = init_weight((hidden_feats, in_feats))
    wi = init_weight((hidden_feats, in_feats))
    wo = init_weight((hidden_feats, in_feats))
    wc = init_weight((hidden_feats, in_feats))
    uf = init_weight((hidden_feats, hidden_feats))
    ui = init_weight((hidden_feats, hidden_feats))
    uo = init_weight((hidden_feats, hidden_feats))
    uc = init_weight((hidden_feats, hidden_feats))
    bf = init_norm((hidden_feats,))
    bi = init_norm((hidden_feats,))
    bo = init_norm((hidden_feats,))
    bc = init_norm((hidden_feats,))
    d_y = init_norm((hidden_feats,))

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
