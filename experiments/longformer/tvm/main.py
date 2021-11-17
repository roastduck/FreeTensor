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
if sys.argv[1]=='cpu':
    target_name = 'llvm -libs=mkl -mcpu=core-avx2'
elif sys.argv[1]=='gpu':
    target_name = 'cuda -libs=cublas'
else:
    assert(False)

tuning_rounds=1000

n_heads = 8
seq_len = 10000
feat_len = 512
w = 32
dilation = 4  # counts from 1
dilation_heads = 2

target = tvm.target.Target(target_name)
dtype, itype = 'float32', 'int32'
time_now = datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
log_file=f'ansor.{sys.argv[1]}.{time_now}.json'
print('log file:', log_file)

d_q = np.load("../q.in.npy")
d_k = np.load("../k.in.npy")
d_v = np.load("../v.in.npy")
d_y = np.load("../d_y.in.npy")
# output_shape=()

if True:  # Tune SG2BMM kernel
    @auto_scheduler.register_workload  # Note the auto_scheduler decorator
    def longformer_compute0(n_heads, seq_len, feat_len, w, dilation, dilation_heads, dtype):
        q = te.placeholder((n_heads, seq_len, feat_len), name="q", dtype=dtype)
        k = te.placeholder((n_heads, seq_len, feat_len), name="k", dtype=dtype)
        pad = dilation*w
        k_pad = te.compute(
            (n_heads, seq_len+2*pad, feat_len),
            lambda a, b, c: tvm.tir.if_then_else(
                tvm.tir.all(b >= pad, b - pad < seq_len),
                k[a, b - pad, c],
                tvm.tir.const(0.0, dtype),
            ),
            name="Kpad",
        )

        p = te.reduce_axis((0, feat_len), name="p")
        prob = te.compute(
            (n_heads, seq_len, 2*w+1),
            # TVM unsupport: Reductions are only allowed at the top level of compute
            # lambda i, j, k: tvm.tir.if_then_else(i < dilation_heads,
            #                                      te.sum(
            #                                          q[i, j, p]*k_pad[i, j+dilation*(k-w), p], axis=p),
            #                                      te.sum(q[i, j, p]*k_pad[i, j+(k-w), p], axis=p)),
            lambda i, j, k:
            te.sum(tvm.tir.if_then_else(i < dilation_heads,
                                        q[i, j, p]*k_pad[i, j+dilation*(k-w)+pad, p], 
                                        q[i, j, p]*k_pad[i, j+(k-w)+pad, p]), axis=p),
            name="SG2BMM",
        )
        return [q, k, prob]

    @auto_scheduler.register_workload  # Note the auto_scheduler decorator
    def longformer_compute1(n_heads, seq_len, feat_len, w, dilation, dilation_heads, dtype):
        prob = te.placeholder((n_heads, seq_len, 2*w+1),
                              name="prob", dtype=dtype)
        v = te.placeholder((n_heads, seq_len, feat_len), name="q", dtype=dtype)
        pad = dilation*w
        v_pad = te.compute(
            (n_heads, seq_len+2*pad, feat_len),
            lambda a, b, c: tvm.tir.if_then_else(
                    tvm.tir.all(b >= pad, b - pad < seq_len),
                    v[a, b - pad, c],
                tvm.tir.const(0.0, dtype),
            ),
            name="Qpad",
        )

        p = te.reduce_axis((0, 2*w+1), name="p")
        attn = te.compute(
            (n_heads, seq_len, feat_len),
            lambda i, j, k: te.sum(tvm.tir.if_then_else(i < dilation_heads,
                                        prob[i, j, p]*v_pad[i, j+dilation*(p-w)+pad, k], 
                                        prob[i, j, p]*v_pad[i, j+(p-w)+pad, k]), axis=p),
            name="G2BMM",
        )
        return [prob, v, attn]

    @auto_scheduler.register_workload
    def softmax_layer(n_heads, seq_len, feat_len, w, dtype):
        x = tvm.te.placeholder((n_heads, seq_len, 2 * w + 1), name='x', dtype=dtype)
        out = topi.nn.softmax(x)/(math.sqrt(feat_len))
        return [x, out]


    # Only dilated parts
    @auto_scheduler.register_workload  # Note the auto_scheduler decorator
    def SG2BMM_compute(n_heads, seq_len, feat_len, w, dilation, dtype):
        q = te.placeholder((n_heads, seq_len, feat_len), name="q", dtype=dtype)
        k = te.placeholder((n_heads, seq_len, feat_len), name="k", dtype=dtype)
        v = te.placeholder((n_heads, seq_len, feat_len), name="v", dtype=dtype)
        pad = dilation*w
        k_pad = te.compute(
            (n_heads, seq_len+2*pad, feat_len),
            lambda a, b, c: tvm.tir.if_then_else(
                tvm.tir.all(b >= pad, b - pad < seq_len),
                k[a, b - w, c],
                tvm.tir.const(0.0, dtype),
            ),
            name="Kpad",
        )

        p = te.reduce_axis((0, feat_len), name="p")
        sum = te.compute(
            (n_heads, seq_len, 2*w+1),
            lambda i, j, k: te.sum(
                q[i, j, p]*k_pad[i, j+dilation*(k-w), p], axis=p),
            name="SG2BMM",
        )
        return [q, k, v, sum]

    @auto_scheduler.register_workload  # Note the auto_scheduler decorator
    def G2BMM_compute(n_heads, seq_len, feat_len, w, dilation, dtype):
        prob = te.placeholder((n_heads, seq_len, 2*w+1),
                              name="prob", dtype=dtype)
        q = te.placeholder((n_heads, seq_len, feat_len), name="q", dtype=dtype)
        pad = dilation*w
        q_pad = te.compute(
            (n_heads, seq_len+2*pad, feat_len),
            lambda a, b, c: tvm.tir.if_then_else(
                    tvm.tir.all(b >= pad, b - pad < seq_len),
                    q[a, b - w, c],
                tvm.tir.const(0.0, dtype),
            ),
            name="Qpad",
        )

        p = te.reduce_axis((0, 2*w+1), name="p")
        sum = te.compute(
            (n_heads, seq_len, feat_len),
            lambda i, j, k: te.sum(
                prob[i, j, p]*q_pad[i, j+dilation*(p-w), k], axis=p),
            name="G2BMM",
        )
        return [prob, q, sum]

    ################################################################################
    tasks = [
        tvm.auto_scheduler.SearchTask(
            func=longformer_compute0, args=(
                n_heads, seq_len, feat_len, w, dilation, dilation_heads, dtype),
            target=target),
        tvm.auto_scheduler.SearchTask(
            func=softmax_layer, args=(
                n_heads, seq_len, feat_len, w, dtype),
            target=target),
        tvm.auto_scheduler.SearchTask(
            func=longformer_compute1, args=(
                n_heads, seq_len, feat_len, w, dilation, dilation_heads, dtype),
            target=target),

        # tvm.auto_scheduler.SearchTask(
        # func=SG2BMM_compute, args=(
        #     n_heads, seq_len, feat_len, w, dilation, dtype),
        # target=target)
        # tvm.auto_scheduler.SearchTask(
        # func=G2BMM_compute, args=(
        #     n_heads, seq_len, feat_len, w, dilation, dtype),
        # target=target)
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

    funcs=[]
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
    q_tvm = tvm.nd.array(d_q, device=dev)
    k_tvm = tvm.nd.array(d_k, device=dev)
    v_tvm = tvm.nd.array(d_v, device=dev)
    prob_tvm = tvm.nd.empty((n_heads, seq_len, 2*w+1), device=dev)
    prob_softmax_tvm = tvm.nd.empty((n_heads, seq_len, 2*w+1), device=dev)
    y_tvm = tvm.nd.empty((n_heads, seq_len, feat_len), device=dev)
    # funcs[0](q_tvm, k_tvm, prob_tvm)
    # funcs[1](prob_tvm, prob_softmax_tvm)
    # funcs[2](prob_softmax_tvm, v_tvm, y_tvm)

    inputs_funcs = [(q_tvm, k_tvm, prob_tvm), (prob_tvm, prob_softmax_tvm),
            (prob_softmax_tvm, v_tvm, y_tvm)]
    # Check correctness
    for func, inputs in zip(funcs, inputs_funcs):
        func(*inputs)
    np.save('y.out.npy', y_tvm.asnumpy())

    # Evaluation
    warmup_num = 10
    timing_repeat = 1000
    time_log=[]
    for func, inputs in zip(funcs, inputs_funcs):
        evaluator = func.time_evaluator(func.entry_name, dev, number=warmup_num)
        evaluator(*inputs)
        evaluator = func.time_evaluator(func.entry_name, dev, number=timing_repeat)
        time_ms=np.median(evaluator(*inputs).results) * 1000
        time_log.append(time_ms)
    print(f"{warmup_num} warmup, {timing_repeat} repeats for evalution")
    print('Time breakdown (ms):', time_log)
    print(
        "Average e2e time: %.3f ms"
        % (sum(time_log))
    )
