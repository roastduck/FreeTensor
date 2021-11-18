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
elif sys.argv[1] == 'gpu':
    target_name = 'cuda -libs=cublas'
else:
    assert(False)

tuning_rounds = 1000

# vertices = np.load("../vertices.in.npy").astype("float32")
# y = np.zeros((n_faces, h, w), dtype="float32")

if target_name.startswith('llvm'):
    dev = tvm.cpu()
elif target_name.startswith('cuda'):
    dev = tvm.cuda()
else:
    assert False
# vertices_tvm = tvm.nd.array(vertices, device=dev)

target = tvm.target.Target(target_name)
dtype, itype = 'float32', 'int32'
time_now = datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
log_file = f'ansor.{sys.argv[1]}.{time_now}.json'
print('log file:', log_file)

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def get_lstm_compute():
    num_thread_y = 8
    num_thread_x = 16 * 3 // 2
    num_sm = 24
    n_num_step = 128
    num_step = 1000
    num_hidden = 1152 // 2
    batch_size = 1
    # Global transition matrix
    # Input hidden channel can be pre-caculated by a gemm
    Xi2h = te.placeholder((num_step, batch_size, 4, num_hidden), name="Xi2h")
    # Only handle hidden transition, saves space.
    Wh2h = te.placeholder((4, num_hidden, num_hidden), name="Wh2h")
    # h: output hidden state, c: cell state.
    s_state_h = te.placeholder((num_step, batch_size, num_hidden))
    s_state_c = te.placeholder((num_step, batch_size, num_hidden))
    s_init_c = te.compute((1, batch_size, num_hidden), lambda *i: 0.0, name="init_c")
    s_init_h = te.compute((1, batch_size, num_hidden), lambda *i: 0.0, name="init_h")
    # LSTM transition
    k = te.reduce_axis((0, num_hidden), name="ki2h")
    s_h2h = te.compute(
        (num_step, batch_size, 4, num_hidden),
        lambda t, i, x, j: te.sum(s_state_h[t - 1, i, k] * Wh2h[x, j, k], axis=k),
        name="s_h2h",
    )
    # Gate rules
    gates = te.compute(Xi2h.shape, lambda *i: Xi2h(*i) + s_h2h(*i), name="gates")
    gshape = (num_step, batch_size, num_hidden)
    in_gate = te.compute(gshape, lambda t, i, j: te.sigmoid(gates[t, i, 0, j]), name="in_gate")
    in_transform = te.compute(
        gshape, lambda t, i, j: te.tanh(gates[t, i, 1, j]), name="in_transform"
    )
    forget_gate = te.compute(
        gshape, lambda t, i, j: te.sigmoid(gates[t, i, 2, j]), name="forget_gate"
    )
    out_gate = te.compute(gshape, lambda t, i, j: te.sigmoid(gates[t, i, 3, j]), name="out_gate")
    next_c = te.compute(
        gshape,
        lambda t, i, j: forget_gate[t, i, j] * s_state_c[t - 1, i, j]
        + in_gate[t, i, j] * in_transform[t, i, j],
        name="next_c",
    )
    next_h = te.compute(
        gshape, lambda t, i, j: out_gate[t, i, j] * te.tanh(next_c[t, i, j]), name="next_h"
    )
    update_c = te.compute(gshape, lambda *i: next_c(*i), name="update_c")
    update_h = te.compute(gshape, lambda *i: next_h(*i), name="update_h")
    # schedule
    scan_h, scan_c = tvm.te.scan(
        [s_init_h, s_init_c],
        [update_h, update_c],
        [s_state_h, s_state_c],
        inputs=[Xi2h],
        name="lstm_scan",
    )
    return [Xi2h, Wh2h, scan_h, scan_c]

# Only dilated parts
args = get_lstm_compute()
s = te.create_schedule(args[-1].op)
# print(tvm.lower(s, args, simple_mode=True))
# y_tvm = tvm.nd.empty(y.shape, device=dev)


# if sys.argv[1] == 'cpu':
#     func = tvm.build(s, args, target)
#     func(vertices_tvm, faces_tvm, y_tvm)
#     func(v_tvm, y_tvm)
#     np.save('y.out.npy', y_tvm.numpy())

# optimized = (
#     np.array(timeit.Timer(lambda:func(vertices_tvm, faces_tvm, y_tvm)).repeat(
#         repeat=10, number=1))
#     * 1000 / 1
# )
# print(optimized)

################################################################################
tasks = [
    tvm.auto_scheduler.SearchTask(
        func=get_lstm_compute, args=(
            ),
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

if sys.argv[1]=='cpu':
    inputs_funcs = [
        # (vertices_tvm, faces_tvm, v_tvm),
        (v_tvm, y_tvm)]
else:
    inputs_funcs = [
        (vertices_tvm, faces_tvm, v_tvm),
        (v_tvm, e_cp_tvm, dist_tvm),
        (e_cp_tvm, dist_tvm,y_tvm),
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
