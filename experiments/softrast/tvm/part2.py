import timeit
import time
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
import argparse
# Enable debug logs
import logging

sys.path.append('../..')
from common.numpy.io import load_txt, store_txt

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('target', nargs='?')
parser.add_argument('--tune', action='store_true', dest='is_tuning')
parser.add_argument('--tune-rounds',
                    type=int,
                    default=1000,
                    dest='tuning_rounds')
parser.add_argument('--eval', default='', dest='eval')
parser.add_argument('--warmup-repeat', type=int, default=10, dest='warmup_num')
parser.add_argument('--timing-repeat', type=int, default=100, dest='test_num')
parser.add_argument('--profile-gpu', action='store_true', dest='profile_gpu')
cmd_args = parser.parse_args()

if cmd_args.profile_gpu:
    from common.gpu import profile_start, profile_stop

if cmd_args.target == 'cpu':
    target_name = 'llvm -libs=mkl -mcpu=core-avx2'
elif cmd_args.target == 'gpu':
    target_name = 'cuda -libs=cublas'
else:
    assert False
if cmd_args.is_tuning:
    time_now = datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
    log_file = f'ansor.{cmd_args.target}.{time_now}.json'
else:
    log_file = cmd_args.eval
    if log_file == '':
        print("Please specify --eval <log_file> if not tuning")
        exit(-1)

vertices = load_txt("../vertices.in", "float32")
faces = load_txt("../faces.in", "int32")
v = load_txt("v.tmp", "float32")
n_verts = vertices.shape[0]
n_faces = faces.shape[0]
h = 64
w = 64
y = np.zeros((n_faces, h, w), dtype="float32")

if target_name.startswith('llvm'):
    dev = tvm.cpu()
elif target_name.startswith('cuda'):
    dev = tvm.cuda()
else:
    assert False
vertices_tvm = tvm.nd.array(vertices, device=dev)
faces_tvm = tvm.nd.array(faces, device=dev)
v_tvm = tvm.nd.array(v, device=dev)

target = tvm.target.Target(target_name)
dtype, itype = 'float32', 'int32'
print('log file:', log_file)

print(n_verts, n_faces)


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def contiguous_compute(n_verts,
                       n_faces,
                       h,
                       w,
                       sigma=1e-4,
                       dtype='float32',
                       itype='int32'):
    vertices = te.placeholder((n_verts, 3), name="vertices", dtype=dtype)
    faces = te.placeholder((n_faces, 3), name="faces", dtype=itype)
    # v = topi.adv_index(vertices, [faces, topi.const_vector([0, 1])])
    v = topi.adv_index(vertices, [faces])
    return [vertices, faces, v]


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def softrast_compute(n_verts,
                     n_faces,
                     h,
                     w,
                     sigma=1e-4,
                     dtype='float32',
                     itype='int32'):
    # # Indirect memory acess causes autoscheduler failure
    # vertices = te.placeholder((n_verts, 3), name="vertices", dtype=dtype)
    # faces = te.placeholder((n_faces, 3), name="faces", dtype=itype)
    # # v = topi.adv_index(vertices, [faces, topi.const_vector([0, 1])])
    # v = topi.adv_index(vertices, [faces])

    # v[face][vectex][x/y-axis]
    v = te.placeholder((n_faces, 3, 3), name="v", dtype=dtype)

    e_cp = te.compute(
        (n_faces, h, w, 3),
        lambda i, j, k, p: ((1. / (h - 1) * j) - v[i, p, 0]) *
        (v[i, (p + 1) % 3, 1] - v[i, p, 1]) -
        ((1. / (w - 1) * k) - v[i, p, 1]) * (v[i, (p + 1) % 3, 0] - v[i, p, 0]),
        name="e_cp",
    )
    # p0 = te.reduce_axis((0, 3), name="p")
    # inside = te.compute(
    #     (n_faces, h, w),
    #     lambda i, j, k:
    #         # te.if_then_else(te.all(e_cp[i, j, k, 0] < 0, e_cp[i, j, k, 1] < 0, e_cp[i, j, k, 2] < 0), 1.0,0.0),
    #         te.all(e_cp[i, j, k, 0] < 0, e_cp[i, j, k, 1] < 0, e_cp[i, j, k, 2] < 0),
    #     name="inside",
    # )

    p1 = te.reduce_axis((0, 3), name="p")
    dist = te.compute(
        (n_faces, h, w),
        lambda i, j, k: te.min(
            te.if_then_else(
                # L78 if dp1 >= 0
                ((1. / (h - 1) * j) - v[i, p1, 0]) *
                (v[i, (p1 + 1) % 3, 0] - v[i, p1, 0]) + (
                    (1. / (w - 1) * k) - v[i, p1, 1]) *
                (v[i, (p1 + 1) % 3, 1] - v[i, p1, 1]) >= 0,
                # if dp2 >= 0
                te.if_then_else(
                    ((1. / (h - 1) * j) - v[i, (p1 + 1) % 3, 0]) *
                    (v[i, p1, 0] - v[i, (p1 + 1) % 3, 0]) + (
                        (1. / (w - 1) * k) - v[i, (p1 + 1) % 3, 1]) *
                    (v[i, p1, 1] - v[i, (p1 + 1) % 3, 1]) >= 0,
                    # L82: dp2>=0
                    te.abs(e_cp[i, j, k, p1]) / te.sqrt(
                        (v[i, (p1 + 1) % 3, 0] - v[i, p1, 0]) *
                        (v[i, (p1 + 1) % 3, 0] - v[i, p1, 0]) +
                        (v[i, (p1 + 1) % 3, 1] - v[i, p1, 1]) *
                        (v[i, (p1 + 1) % 3, 1] - v[i, p1, 1])),
                    # L85
                    te.sqrt(((1. / (h - 1) * j) - v[i, (p1 + 1) % 3, 0]) * (
                        (1. / (h - 1) * j) - v[i, (p1 + 1) % 3, 0]) +
                            ((1. / (w - 1) * k) - v[i, (p1 + 1) % 3, 1]) * (
                                (1. / (w - 1) * k) - v[i, (p1 + 1) % 3, 1]))),
                # L88
                te.sqrt(((1. / (h - 1) * j) - v[i, p1, 0]) *
                        ((1. / (h - 1) * j) - v[i, p1, 0]) + (
                            (1. / (w - 1) * k) - v[i, p1, 1]) * (
                                (1. / (w - 1) * k) - v[i, p1, 1]))),
            axis=p1),
        name='dist')
    # L97
    y = te.compute((n_faces, h, w),
                   lambda i, j, k: te.sigmoid(
                       te.if_then_else(
                           te.all(e_cp[i, j, k, 0] < 0, e_cp[i, j, k, 1] < 0,
                                  e_cp[i, j, k, 2] < 0), 1, -1) * dist[i, j, k]
                       * dist[i, j, k] / sigma),
                   name='y')
    # return [vertices, faces, y]
    return [v, y]


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def softrast_gpu_compute1(n_verts,
                          n_faces,
                          h,
                          w,
                          sigma=1e-4,
                          dtype='float32',
                          itype='int32'):
    # v[face][vectex][x/y-axis]
    v = te.placeholder((n_faces, 3, 3), name="v", dtype=dtype)

    e_cp = te.compute(
        (n_faces, h, w, 3),
        lambda i, j, k, p: ((1. / (h - 1) * j) - v[i, p, 0]) *
        (v[i, (p + 1) % 3, 1] - v[i, p, 1]) -
        ((1. / (w - 1) * k) - v[i, p, 1]) * (v[i, (p + 1) % 3, 0] - v[i, p, 0]),
        name="e_cp",
    )

    p1 = te.reduce_axis((0, 3), name="p")
    dist = te.compute(
        (n_faces, h, w),
        lambda i, j, k: te.min(
            te.if_then_else(
                # L78 if dp1 >= 0
                ((1. / (h - 1) * j) - v[i, p1, 0]) *
                (v[i, (p1 + 1) % 3, 0] - v[i, p1, 0]) + (
                    (1. / (w - 1) * k) - v[i, p1, 1]) *
                (v[i, (p1 + 1) % 3, 1] - v[i, p1, 1]) >= 0,
                # if dp2 >= 0
                te.if_then_else(
                    ((1. / (h - 1) * j) - v[i, (p1 + 1) % 3, 0]) *
                    (v[i, p1, 0] - v[i, (p1 + 1) % 3, 0]) + (
                        (1. / (w - 1) * k) - v[i, (p1 + 1) % 3, 1]) *
                    (v[i, p1, 1] - v[i, (p1 + 1) % 3, 1]) >= 0,
                    # L82: dp2>=0
                    te.abs(e_cp[i, j, k, p1]) / te.sqrt(
                        (v[i, (p1 + 1) % 3, 0] - v[i, p1, 0]) *
                        (v[i, (p1 + 1) % 3, 0] - v[i, p1, 0]) +
                        (v[i, (p1 + 1) % 3, 1] - v[i, p1, 1]) *
                        (v[i, (p1 + 1) % 3, 1] - v[i, p1, 1])),
                    # L85
                    te.sqrt(((1. / (h - 1) * j) - v[i, (p1 + 1) % 3, 0]) * (
                        (1. / (h - 1) * j) - v[i, (p1 + 1) % 3, 0]) +
                            ((1. / (w - 1) * k) - v[i, (p1 + 1) % 3, 1]) * (
                                (1. / (w - 1) * k) - v[i, (p1 + 1) % 3, 1]))),
                # L88
                te.sqrt(((1. / (h - 1) * j) - v[i, p1, 0]) *
                        ((1. / (h - 1) * j) - v[i, p1, 0]) + (
                            (1. / (w - 1) * k) - v[i, p1, 1]) * (
                                (1. / (w - 1) * k) - v[i, p1, 1]))),
            axis=p1),
        name='dist')
    return [v, e_cp, dist]


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def softrast_gpu_compute2(n_verts,
                          n_faces,
                          h,
                          w,
                          sigma=1e-4,
                          dtype='float32',
                          itype='int32'):
    e_cp = te.placeholder((n_faces, h, w, 3), name="e_cp", dtype=dtype)
    dist = te.placeholder((n_faces, h, w), name="dist", dtype=dtype)

    # L97
    y = te.compute((n_faces, h, w),
                   lambda i, j, k: te.sigmoid(
                       te.if_then_else(
                           te.all(e_cp[i, j, k, 0] < 0, e_cp[i, j, k, 1] < 0,
                                  e_cp[i, j, k, 2] < 0), 1, -1) * dist[i, j, k]
                       * dist[i, j, k] / sigma),
                   name='y')
    # return [vertices, faces, y]
    return [e_cp, dist, y]


# Only dilated parts
#args1 = softrast_gpu_compute1(n_verts, n_faces, h, w)
#args2 = softrast_gpu_compute2(n_verts, n_faces, h, w)
#s1 = te.create_schedule(args1[-1].op)
#s2 = te.create_schedule(args2[-1].op)
# print(tvm.lower(s, args, simple_mode=True))
y_tvm = tvm.nd.empty(y.shape, device=dev)
#v_tvm = tvm.nd.empty((n_faces, 3, 3), device=dev)
inside_tvm = tvm.nd.empty((n_faces, h, w), dtype='bool', device=dev)
e_cp_tvm = tvm.nd.empty((n_faces, h, w, 3), device=dev)
dist_tvm = tvm.nd.empty((n_faces, h, w), device=dev)

#if cmd_args.target == 'cpu':
#    func1 = tvm.build(s1, args1, target)
#    func2 = tvm.build(s2, args2, target)
#    func1(v_tvm, e_cp_tvm, dist_tvm)
#    func2(e_cp_tvm, dist_tvm, y_tvm)
#    store_txt('y.out', y_tvm.numpy())
#    exit(-1)

# optimized = (
#     np.array(timeit.Timer(lambda:func(vertices_tvm, faces_tvm, y_tvm)).repeat(
#         repeat=10, number=1))
#     * 1000 / 1
# )
# print(optimized)

################################################################################
if cmd_args.target == 'cpu':
    tasks = [
        # tvm.auto_scheduler.SearchTask(
        #     func=contiguous_compute, args=(
        #         n_verts,n_faces,h,w),
        #     target=target),
        tvm.auto_scheduler.SearchTask(func=softrast_compute,
                                      args=(n_verts, n_faces, h, w),
                                      target=target),
    ]
else:
    tasks = [
        #tvm.auto_scheduler.SearchTask(func=contiguous_compute,
        #                              args=(n_verts, n_faces, h, w),
        #                              target=target),
        tvm.auto_scheduler.SearchTask(func=softrast_gpu_compute1,
                                      args=(n_verts, n_faces, h, w),
                                      target=target),
        tvm.auto_scheduler.SearchTask(func=softrast_gpu_compute2,
                                      args=(n_verts, n_faces, h, w),
                                      target=target),
    ]

################################################################################
# Set Parameters for Auto-Scheduler
if cmd_args.is_tuning:

    print()
    print("!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!! TUNING             !!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!")
    print()

    tuner = auto_scheduler.TaskScheduler(tasks)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=cmd_args.tuning_rounds * len(tasks),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )

    st_tune = time.time()
    tuner.tune(tune_option)
    en_tune = time.time()
    print(f"Tuning time: {en_tune - st_tune}s")

print()
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print("!! EVALUATING PART 2  !!")
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print()

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

if cmd_args.target == 'cpu':
    inputs_funcs = [
        # (vertices_tvm, faces_tvm, v_tvm),
        (v_tvm, y_tvm)
    ]
else:
    inputs_funcs = [
        #(vertices_tvm, faces_tvm, v_tvm),
        (v_tvm, e_cp_tvm, dist_tvm),
        (e_cp_tvm, dist_tvm, y_tvm),
    ]
# Check correctness
for func, inputs in zip(funcs, inputs_funcs):
    func(*inputs)
store_txt('y.out', y_tvm.asnumpy())

# Evaluation
time_log = []
for func, inputs in zip(funcs, inputs_funcs):
    evaluator = func.time_evaluator(func.entry_name,
                                    dev,
                                    number=cmd_args.warmup_num)
    evaluator(*inputs)
if cmd_args.profile_gpu:
    profile_start()
for func, inputs in zip(funcs, inputs_funcs):
    evaluator = func.time_evaluator(func.entry_name,
                                    dev,
                                    number=cmd_args.test_num)
    time_ms = np.median(evaluator(*inputs).results) * 1000
    time_log.append(time_ms)
if cmd_args.profile_gpu:
    profile_stop()
print(
    f"{cmd_args.warmup_num} warmup, {cmd_args.test_num} repeats for evalution")
print('Time breakdown (ms):', time_log)
print("Average e2e time: %.3f ms" % (sum(time_log)))
