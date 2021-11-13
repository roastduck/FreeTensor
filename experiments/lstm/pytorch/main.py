import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch._VF as vf

def nn_lstm(x, lstm_layer, h, c):
    # TODO: Dilation
    # TODO: Stride
    # TODO: Batch
    length = x.shape[0]
    in_feats = x.shape[1]
    hidden_feats = 256
    n_layers = 1

    lstm_layer = nn.LSTM(in_feats, hidden_feats, n_layers, batch_first=True)
    batch_size = 1


    inp = x.reshape((1, length, in_feats))
    hidden_state = torch.zeros(n_layers, batch_size, hidden_feats)
    cell_state = torch.zeros(n_layers, batch_size, hidden_feats)
    hidden = (hidden_state, cell_state)
    out, hidden = lstm_layer(inp, hidden)
    out = out.squeeze()[-1, :]
    return out

def lstm(x, wi, ui, bi, wf, uf, bf, wc, uc, bc, wo, uo, bo, h, c):
    length = x.shape[0]

    for k in range(length):
        f = torch.sigmoid(wf @ x[k] + uf @ h + bf)
        i = torch.sigmoid(wi @ x[k] + ui @ h + bi)
        o = torch.sigmoid(wo @ x[k] + uo @ h + bo)
        cc = torch.tanh(wc @ x[k] + uc @ h + bc)
        c = f * c + i * cc
        h = o * torch.tanh(c)

    return h


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu/gpu>")
        exit(-1)
    device = sys.argv[1]
    hidden_feats = 256
    x = torch.tensor(np.loadtxt("../x.in"), dtype=torch.float)
    d_y = torch.tensor(np.loadtxt("../d_y.in"), dtype=torch.float)
    wi = torch.tensor(np.loadtxt("../wi.in"), dtype=torch.float)
    wc = torch.tensor(np.loadtxt("../wc.in"), dtype=torch.float)
    wf = torch.tensor(np.loadtxt("../wf.in"), dtype=torch.float)
    wo = torch.tensor(np.loadtxt("../wo.in"), dtype=torch.float)
    ui = torch.tensor(np.loadtxt("../ui.in"), dtype=torch.float)
    uc = torch.tensor(np.loadtxt("../uc.in"), dtype=torch.float)
    uf = torch.tensor(np.loadtxt("../uf.in"), dtype=torch.float)
    uo = torch.tensor(np.loadtxt("../uo.in"), dtype=torch.float)
    bi = torch.tensor(np.loadtxt("../bi.in"), dtype=torch.float)
    bc = torch.tensor(np.loadtxt("../bc.in"), dtype=torch.float)
    bf = torch.tensor(np.loadtxt("../bf.in"), dtype=torch.float)
    bo = torch.tensor(np.loadtxt("../bo.in"), dtype=torch.float)
    h = torch.zeros(hidden_feats, dtype=torch.float)
    c = torch.zeros(hidden_feats, dtype=torch.float)

    if device == 'gpu':
        x = x.cuda()
        wi = wi.cuda()
        wc = wc.cuda()
        wf = wf.cuda()
        wo = wo.cuda()
        ui = ui.cuda()
        uc = uc.cuda()
        uf = uf.cuda()
        uo = uo.cuda()
        bi = bi.cuda()
        bc = bc.cuda()
        bf = bf.cuda()
        bo = bo.cuda()
        d_y = d_y.cuda()
        h = h.cuda()
        c = c.cuda()
        sync = torch.cuda.synchronize
    else:
        assert device == 'cpu'
        sync = lambda: None

    warmup_num = 10
    test_num = 1000

    for i in range(warmup_num):
        y = lstm(x, wi, ui, bi, wf, uf, bf, wc, uc, bc, wo, uo, bo, h, c)
        if i == 0:
            np.savetxt("y.out", y.cpu().detach().numpy())
    sync()
    t0 = time.time()
    for i in range(test_num):
        y = lstm(x, wi, ui, bi, wf, uf, bf, wc, uc, bc, wo, uo, bo, h, c)
    sync()
    t1 = time.time()
    assert y.shape == (hidden_feats, )
    print(f"Pytorch Inference Time = {(t1 - t0) / test_num * 1000} ms")

    x.requires_grad = True
    wi.requires_grad = True
    wc.requires_grad = True
    wf.requires_grad = True
    wo.requires_grad = True
    ui.requires_grad = True
    uc.requires_grad = True
    uf.requires_grad = True
    uo.requires_grad = True
    bi.requires_grad = True
    bc.requires_grad = True
    bf.requires_grad = True
    bo.requires_grad = True
    for i in range(warmup_num):
        y = lstm(x, wi, ui, bi, wf, uf, bf, wc, uc, bc, wo, uo, bo, h, c)
    sync()
    t0 = time.time()
    for i in range(test_num):
        y = lstm(x, wi, ui, bi, wf, uf, bf, wc, uc, bc, wo, uo, bo, h, c)
    sync()
    t1 = time.time()
    assert y.shape == (hidden_feats,)
    print(f"Pytorch Forward Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        y.backward(d_y, retain_graph=True)
        if i == 0:
            np.savetxt("d_x.out", x.grad.cpu().numpy())
            np.savetxt("d_wi.out", wi.grad.cpu().numpy())
            np.savetxt("d_wc.out", wc.grad.cpu().numpy())
            np.savetxt("d_wf.out", wf.grad.cpu().numpy())
            np.savetxt("d_wo.out", wo.grad.cpu().numpy())
            np.savetxt("d_ui.out", ui.grad.cpu().numpy())
            np.savetxt("d_uc.out", uc.grad.cpu().numpy())
            np.savetxt("d_uf.out", uf.grad.cpu().numpy())
            np.savetxt("d_uo.out", uo.grad.cpu().numpy())
            np.savetxt("d_bi.out", bi.grad.cpu().numpy())
            np.savetxt("d_bc.out", bc.grad.cpu().numpy())
            np.savetxt("d_bf.out", bf.grad.cpu().numpy())
            np.savetxt("d_bo.out", bo.grad.cpu().numpy())
    sync()
    t0 = time.time()
    for i in range(test_num):
        y.backward(d_y, retain_graph=True)
    sync()
    t1 = time.time()
    print(f"Pytorch Backward Time = {(t1 - t0) / test_num * 1000} ms")
