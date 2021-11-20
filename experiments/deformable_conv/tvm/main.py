import timeit
import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing
import math

import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import topi, autotvm
import logging
from datetime import datetime
import sys
# Enable debug logs
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <cpu/gpu>")
    exit(-1)
if sys.argv[1] == 'cpu':
    target_name = 'llvm -libs=mkl -mcpu=core-avx2'
    dev = tvm.cpu()
elif sys.argv[1] == 'gpu':
    target_name = 'cuda -libs=cublas'
    dev = tvm.cuda()
else:
    assert(False)

tuning_rounds = 1000

target = tvm.target.Target(target_name)
dtype, itype = 'float32', 'int32'
time_now = datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
log_file = f'ansor.{sys.argv[1]}.{time_now}.json'
print('log file:', log_file)

n = 8
c_in = 256
c_out = 256
h = 56
w = 56
k_h = 3
k_w = 3

x = np.random.random([n, c_in, h, w]).astype(dtype) * 2 - 1
w1 = np.random.random([k_h, k_w, 2, c_in, k_h, k_w]).astype(dtype) * 2 - 1
w2 = np.random.random([c_out, c_in, k_h, k_w]).astype(dtype) * 2 - 1
x = np.load('../x.in.npy')
w1 = np.load('../w1.in.npy')
w2 = np.load('../w2.in.npy')

pos_shape = (n, h, w, 3, 3, 2, 2)
pos_size = np.product(pos_shape)


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def deformable_conv_compute0(n, c_in, c_out, h, w, k_h, k_w, dtype='float32', itype='int32'):
    X = te.placeholder((n, c_in, h, w), name="X", dtype=dtype)
    W1 = te.placeholder((k_h, k_w, 2, c_in, k_h, k_w), name="W1", dtype=dtype)

    ki = te.reduce_axis((0, c_in), name="ki")
    ri = te.reduce_axis((0, 3), name="ri")
    si = te.reduce_axis((0, 3), name="si")
    pos = te.compute(  # [n,h,w,kernel_h,kernel_w,x/y]
        (n, h, w, 3, 3, 2),
        lambda i, p, q, ro, so, t:
        # row[ro, so] += X[i, ki, p + ri, q + si] * W1[ro, so, 0, ki, ri, si]
        # col[ro, so] += X[i, ki, p + ri, q + si] * W1[ro, so, 1, ki, ri, si]
        te.sum(X[i, ki, p + ri, q + si] * W1[ro, so, t, ki, ri, si],
               axis=[ki, ri, si]),
        name="pos"
    )
    pos_x = te.compute(
        (n, h, w, 3, 3, 2, 2),
        lambda i, p, q, ro, so, tx, ty:
        topi.cast(te.floor(pos[i, p, q, ro, so, 0]+tx), dtype=itype),
        name="pos_x"
    )
    pos_y = te.compute(
        (n, h, w, 3, 3, 2, 2),
        lambda i, p, q, ro, so, tx, ty:
        topi.cast(te.floor(pos[i, p, q, ro, so, 1]+ty), dtype=itype),
        name="pos_y"
    )
    return [X, W1, pos, pos_x, pos_y]


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def contiguous_compute(n, c_in, c_out, h, w, k_h, k_w, dtype='float32', itype='int32'):
    X = te.placeholder((n, c_in, h, w), name="X", dtype=dtype)
    pos_n = te.placeholder(pos_shape, name="pos_n", dtype=dtype)
    pos_x = te.placeholder(pos_shape, name="pos_x", dtype=dtype)
    pos_y = te.placeholder(pos_shape, name="pos_y", dtype=dtype)
    X_hwnc = topi.transpose(X, [2, 3, 0, 1])  # [h, w, n, c_in]
    pos_n_vec = topi.reshape(pos_n, [pos_size])
    pos_x_vec = topi.reshape(pos_x, [pos_size])
    pos_y_vec = topi.reshape(pos_y, [pos_size])
    v_xy = topi.adv_index(X_hwnc, [pos_x_vec, pos_y_vec, pos_n_vec])
    v_xy = topi.reshape(v_xy, [n, h, w, 3, 3, 2, 2, c_in])
    v_xy = topi.transpose(v_xy, [0, 7, 1, 2, 3, 4, 5, 6])
    return [X, pos_n, pos_x, pos_y, v_xy]


# Cannot be tuned by Ansor.
@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def deformable_conv_compute1(n, c_in, c_out, h, w, k_h, k_w, dtype='float32', itype='int32'):
    # [n,h,w,kernel_h,kernel_w,x/y]
    pos = te.placeholder((n, h, w, 3, 3, 2), name="pos", dtype=dtype)
    v_xy = te.placeholder((n, c_in, h, w, 3, 3, 2, 2),
                          name="v_xy", dtype=dtype)
    W2 = te.placeholder((c_out, c_in, k_h, k_w), name="W2", dtype=dtype)

    ci = te.reduce_axis((0, c_in), name="ci")  # channel in
    kx = te.reduce_axis((0, 2), name="kx")  # x axis for 2x2 block
    ky = te.reduce_axis((0, 2), name="ky")  # y axis for 2x2 block
    ri = te.reduce_axis((0, 3), name="ri")  # x axis for 3x3 kernel
    si = te.reduce_axis((0, 3), name="si")  # y axis for 3x3 kernel

    def y_compute(i, ko, p, q):
        def xx():
            return pos[i, p, q, ri, si, 0]

        def yy():
            return pos[i, p, q, ri, si, 1]
        return te.sum(
            te.if_then_else(
                te.all(xx()+kx >= 0, xx()+kx < h, yy()+ky >= 0, yy()+ky < w),
                v_xy[i, ci, p, q, ri, si, kx, ky] *
                te.abs((xx()-te.floor(xx())-kx) * (yy()-te.floor(yy())-ky)),
                0),
            axis=[ci, kx, ky, ri, si])

    y = te.compute(
        (n, c_out, h, w),
        y_compute,
        name='y')
    return [pos, v_xy, W2, y]


# args = deformable_conv_compute1(n, c_in, c_out, h, w, k_h, k_w)
args = contiguous_compute(n, c_in, c_out, h, w, k_h, k_w)
s = te.create_schedule(args[-1].op)
print(tvm.lower(s, args, simple_mode=True))

x_tvm = tvm.nd.array(x, device=dev)
w1_tvm = tvm.nd.array(w1, device=dev)
w2_tvm = tvm.nd.array(w2, device=dev)
pos_tvm = tvm.nd.empty(pos_shape[:-1], device=dev)
pos_n_tvm = tvm.nd.empty(pos_shape, dtype=itype, device=dev)
pos_x_tvm = tvm.nd.empty(pos_shape, dtype=itype, device=dev)
pos_y_tvm = tvm.nd.empty(pos_shape, dtype=itype, device=dev)
v_xy_tvm = tvm.nd.empty((n, c_in, h, w, 3, 3, 2, 2), device=dev)
y_tvm = tvm.nd.empty((n, c_out, h, w), device=dev)

if sys.argv[1] == 'cpu':
    args = deformable_conv_compute0(n, c_in, c_out, h, w, k_h, k_w)
    func = tvm.build(te.create_schedule(args[-1].op), args, target)
    func(x_tvm, w1_tvm, pos_tvm, pos_x_tvm, pos_y_tvm)
    args = contiguous_compute(n, c_in, c_out, h, w, k_h, k_w)
    func = tvm.build(te.create_schedule(args[-1].op), args, target)
    func(x_tvm, pos_n_tvm, pos_x_tvm, pos_y_tvm, v_xy_tvm)
    args = deformable_conv_compute1(n, c_in, c_out, h, w, k_h, k_w)
    func = tvm.build(te.create_schedule(args[-1].op), args, target)
    func(pos_tvm, v_xy_tvm, w2_tvm, y_tvm)

    np.save('y.out.npy', y_tvm.numpy())
exit()

# optimized = (
#     np.array(timeit.Timer(lambda:func(vertices_tvm, faces_tvm, y_tvm)).repeat(
#         repeat=10, number=1))
#     * 1000 / 1
# )
# print(optimized)

################################################################################
tasks = [
    tvm.auto_scheduler.SearchTask(
        func=deformable_conv_compute0, args=(
            n, c_in, c_out, h, w, k_h, k_w),
        # x_tvm, w1_tvm, pos_tvm, pos_x_tvm, pos_y_tvm),
        target=target),
    tvm.auto_scheduler.SearchTask(
        func=contiguous_compute, args=(
            n, c_in, c_out, h, w, k_h, k_w),
        # x_tvm, pos_n_tvm, pos_x_tvm, pos_y_tvm, v_xy_tvm),
        target=target),
    tvm.auto_scheduler.SearchTask(
        func=deformable_conv_compute1, args=(
            n, c_in, c_out, h, w, k_h, k_w),
        # pos_tvm, v_xy_tvm, w2_tvm, y_tvm),
        target=target),
]

################################################################################
# Set Parameters for Auto-Scheduler
tuner = auto_scheduler.TaskScheduler(tasks)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=tuning_rounds*len(tasks),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)
tuner.tune(tune_option)

funcs = []
for task in tasks:
    # Apply the best schedule
    sch, args = task.apply_best(log_file)
    funcs.append(tvm.build(sch, args, target))
# get dev
if target_name.startswith('llvm'):
    dev = tvm.cpu()
elif target_name.startswith('cuda'):
    dev = tvm.cuda()
else:
    assert False
# q_tvm = tvm.nd.array(d_q, device=dev)
# k_tvm = tvm.nd.array(d_k, device=dev)
# v_tvm = tvm.nd.array(d_v, device=dev)
# prob_tvm = tvm.nd.empty((n_heads, seq_len, 2*w+1), device=dev)
# prob_softmax_tvm = tvm.nd.empty((n_heads, seq_len, 2*w+1), device=dev)
# y_tvm = tvm.nd.empty((n_heads, seq_len, feat_len), device=dev)
# funcs[0](q_tvm, k_tvm, prob_tvm)
# funcs[1](prob_tvm, prob_softmax_tvm)
# funcs[2](prob_softmax_tvm, v_tvm, y_tvm)

if sys.argv[1] == 'cpu':
    inputs_funcs = [
        # (vertices_tvm, faces_tvm, v_tvm),
        (v_tvm, y_tvm)]
else:
    inputs_funcs = [
        (vertices_tvm, faces_tvm, v_tvm),
        (v_tvm, e_cp_tvm, dist_tvm),
        (e_cp_tvm, dist_tvm, y_tvm),
    ]
# Check correctness
for func, inputs in zip(funcs, inputs_funcs):
    func(*inputs)
np.save('y.out.npy', y_tvm.asnumpy())

# Evaluation
warmup_num = 10
timing_repeat = 1000
time_log = []
for func, inputs in zip(funcs, inputs_funcs):
    evaluator = func.time_evaluator(
        func.entry_name, dev, number=warmup_num)
    evaluator(*inputs)
    evaluator = func.time_evaluator(
        func.entry_name, dev, number=timing_repeat)
    time_ms = np.median(evaluator(*inputs).results) * 1000
    time_log.append(time_ms)
print(f"{warmup_num} warmup, {timing_repeat} repeats for evalution")
print('Time breakdown (ms):', time_log)
print(
    "Average e2e time: %.3f ms"
    % (sum(time_log))
)
