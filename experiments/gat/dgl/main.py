import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

sys.path.append('../..')
from common.numpy.io import load_txt, store_txt


def load_data(data_name: str):
    '''
    Load data from ../data/{data_name}.config and ../data/{data_name}.graph

    Returns (#vertices, #edges, ptr, idx), where ptr and idx forms a CSR format
    '''

    with open(f"../data/{data_name}.config", 'r') as f:
        num_v, num_e = map(int, f.readline().strip().split(' '))

    with open(f"../data/{data_name}.graph", 'r') as f:
        ptr = np.array(list(map(int, f.readline().strip().split(" "))))
        idx = np.array(list(map(int, f.readline().strip().split(" "))))

    return num_v, num_e, ptr, idx


def gat_layer(g, feat, weight, attn_l, attn_r):
    feat2 = torch.mm(feat, weight)
    att_l = torch.mm(feat2, attn_l.reshape(-1, 1))
    att_r = torch.mm(feat2, attn_r.reshape(-1, 1))
    g.srcdata.update({'ft': feat2, 'el': att_l})
    g.dstdata.update({'er': att_r})
    g.apply_edges(fn.u_add_v('el', 'er', 'e'))
    e = F.leaky_relu(g.edata.pop('e'), 0.1)
    g.edata['a'] = dgl.ops.edge_softmax(g, e)
    g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
    return g.dstdata['ft']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target', nargs='?')
    parser.add_argument('--warmup-repeat',
                        type=int,
                        default=10,
                        dest='warmup_num')
    parser.add_argument('--timing-repeat',
                        type=int,
                        default=100,
                        dest='test_num')
    cmd_args = parser.parse_args()

    device = cmd_args.target

    ptr = load_txt("../ptr.in", "int32")
    idx = load_txt("../idx.in", "int32")
    num_v = ptr.shape[0] - 1
    num_e = idx.shape[0]
    src_list = []
    dst_list = []
    for which in range(num_v):
        for i in range(ptr[which], ptr[which + 1]):
            dst_list.append(which)
            src_list.append(idx[i])
    g = dgl.graph((src_list, dst_list))

    feat_len = 32
    x = torch.tensor(load_txt("../x.in", "float32"), dtype=torch.float)
    w = torch.tensor(load_txt("../w.in", "float32"), dtype=torch.float)
    w_attn_1 = torch.tensor(load_txt("../w_attn_1.in", "float32"),
                            dtype=torch.float)
    w_attn_2 = torch.tensor(load_txt("../w_attn_2.in", "float32"),
                            dtype=torch.float)
    d_y = torch.tensor(load_txt("../d_y.in", "float32"), dtype=torch.float)

    if device == 'gpu':
        x = x.cuda()
        w = w.cuda()
        w_attn_1 = w_attn_1.cuda()
        w_attn_2 = w_attn_2.cuda()
        g = g.to(x.device)
        d_y = d_y.cuda()
        sync = torch.cuda.synchronize
    else:
        assert device == 'cpu'
        sync = lambda: None

    print(
        f"{cmd_args.warmup_num} warmup, {cmd_args.test_num} repeats for evalution"
    )
    warmup_num = cmd_args.warmup_num
    test_num = cmd_args.test_num

    for i in range(warmup_num):
        y = gat_layer(g, x, w, w_attn_1, w_attn_2)
        if i == 0:
            store_txt("y.out", y.cpu().numpy())
    sync()
    t0 = time.time()
    for i in range(test_num):
        y = gat_layer(g, x, w, w_attn_1, w_attn_2)
    sync()
    t1 = time.time()
    assert y.shape == (num_v, feat_len)
    print(f"Inference Time = {(t1 - t0) / test_num * 1000} ms")

    x.requires_grad = True
    w.requires_grad = True
    w_attn_1.requires_grad = True
    w_attn_2.requires_grad = True

    for i in range(warmup_num):
        y = gat_layer(g, x, w, w_attn_1, w_attn_2)
    sync()
    t0 = time.time()
    for i in range(test_num):
        y = gat_layer(g, x, w, w_attn_1, w_attn_2)
    sync()
    t1 = time.time()
    assert y.shape == (num_v, feat_len)
    print(f"Forward Time = {(t1 - t0) / test_num * 1000} ms")

    for i in range(warmup_num):
        y.backward(d_y, retain_graph=True)
        if i == 0:
            store_txt("d_x.out", x.grad.cpu().numpy())
            store_txt("d_w.out", w.grad.cpu().numpy())
            store_txt("d_w_attn_1.out", w_attn_1.grad.cpu().numpy())
            store_txt("d_w_attn_2.out", w_attn_2.grad.cpu().numpy())
    sync()
    t0 = time.time()
    for i in range(test_num):
        y.backward(d_y, retain_graph=True)
    sync()
    t1 = time.time()
    print(f"Backward Time = {(t1 - t0) / test_num * 1000} ms")
