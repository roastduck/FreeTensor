import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


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
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <cpu/gpu> <data_name>")
        exit(-1)
    device = sys.argv[1]
    data_name = sys.argv[2]

    num_v, num_e, ptr, idx = load_data(data_name)
    src_list = []
    dst_list = []
    for which in range(num_v):
        for i in range(ptr[which], ptr[which + 1]):
            dst_list.append(which)
            src_list.append(idx[i])
    g = dgl.graph((src_list, dst_list))

    feat_len = 32
    x = torch.rand(num_v, feat_len, dtype=torch.float)
    w = torch.rand(feat_len, feat_len, dtype=torch.float)
    w_attn_1 = torch.rand(feat_len, dtype=torch.float)
    w_attn_2 = torch.rand(feat_len, dtype=torch.float)

    if device == 'gpu':
        x = x.cuda()
        w = w.cuda()
        w_attn_1 = w_attn_1.cuda()
        w_attn_2 = w_attn_2.cuda()
        g = g.to(x.device)
    else:
        assert device == 'cpu'

    test_num = 1000

    y = gat_layer(g, x, w, w_attn_1, w_attn_2)  # init lazy ops
    t0 = time.time()
    for i in range(test_num):
        y = gat_layer(g, x, w, w_attn_1, w_attn_2)
    t1 = time.time()
    assert y.shape == (num_v, feat_len)
    print(f"Impl1 Time = {(t1 - t0) / test_num * 1000} ms")
