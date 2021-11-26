import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch._VF as vf


def nn_lstm(x, lstm_layer, h, c):
    hidden = (h, c)
    out, hidden = lstm_layer(x, hidden)
    out = out.squeeze()[-1, :]
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu/gpu>")
        exit(-1)
    device = sys.argv[1]
    x = torch.tensor(np.loadtxt("../x.in"), dtype=torch.float)
    d_y = torch.tensor(np.loadtxt("../d_y.in"), dtype=torch.float)
    hidden_feats = d_y.shape[0]
    length = x.shape[0]
    in_feats = x.shape[1]
    n_layers = 1
    batch_size = 1
    x = x.reshape((1, length, in_feats))
    h = torch.zeros(n_layers, batch_size, hidden_feats)
    c = torch.zeros(n_layers, batch_size, hidden_feats)
    lstm_layer = nn.LSTM(in_feats, hidden_feats, n_layers, batch_first=True)
    with torch.no_grad():
        lstm_nograd = nn.LSTM(in_feats,
                              hidden_feats,
                              n_layers,
                              batch_first=True)
    if device == 'gpu':
        x = x.cuda()
        d_y = d_y.cuda()
        h = h.cuda()
        c = c.cuda()
        lstm_layer = lstm_layer.cuda()
        lstm_nograd = lstm_nograd.cuda()
        sync = torch.cuda.synchronize
    else:
        assert device == 'cpu'
        sync = lambda: None

    warmup_num = 10
    test_num = 10
    for i in range(warmup_num):
        with torch.no_grad():
            y = nn_lstm(x, lstm_nograd, h, c)
    sync()
    t0 = time.time()
    for i in range(test_num):
        with torch.no_grad():
            y = nn_lstm(x, lstm_nograd, h, c)
    sync()
    t1 = time.time()
    assert y.shape == (hidden_feats,)
    print(f"Pytorch impl2 Inference Time = {(t1 - t0) / test_num * 1000} ms")
    x.requires_grad = True
    for i in range(warmup_num):
        y = nn_lstm(x, lstm_layer, h, c)
    sync()
    t0 = time.time()
    for i in range(test_num):
        y = nn_lstm(x, lstm_layer, h, c)
    sync()
    t1 = time.time()
    assert y.shape == (hidden_feats,)
    print(f"Pytorch impl2 Forward Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        y.backward(d_y, retain_graph=True)
    sync()
    t0 = time.time()
    for i in range(test_num):
        y.backward(d_y, retain_graph=True)
    sync()
    t1 = time.time()
    print(f"Pytorch impl2 Backward Time = {(t1 - t0) / test_num * 1000} ms")
