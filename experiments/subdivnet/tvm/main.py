import numpy as np
import time

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing

import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
import sys
import argparse
from datetime import datetime
import logging

sys.path.append('../..')
from common.numpy.io import load_txt, store_txt

# Enable debug logs
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

######################################################################
# Define Neural Network in Relay

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

target = tvm.target.Target(target_name)
dtype, itype = 'float32', 'int32'

d_adj = load_txt("../adj.in", itype)
n_faces, in_feats, out_feats = d_adj.shape[0], 13, 64
d_x = load_txt("../x.in", dtype)
d_w = [load_txt(f"../w{i}.in", dtype) for i in range(4)]
d_y = load_txt("../d_y.in", dtype)
output_shape = (n_faces, out_feats)


def get_subdivnet_relay():
    adj = relay.var("adj", shape=(n_faces, 3), dtype=itype)
    x = relay.var("x", shape=(n_faces, in_feats), dtype=dtype)
    w0 = relay.var("w0", shape=(in_feats, out_feats), dtype=dtype)
    w1 = relay.var("w1", shape=(in_feats, out_feats), dtype=dtype)
    w2 = relay.var("w2", shape=(in_feats, out_feats), dtype=dtype)
    w3 = relay.var("w3", shape=(in_feats, out_feats), dtype=dtype)
    sum1 = relay.zeros_like(x)
    sum2 = relay.zeros_like(x)
    sum3 = relay.zeros_like(x)

    adj_flatten = relay.reshape(adj, [n_faces, 3])
    adj_feats = relay.adv_index([x, adj_flatten])
    adj_feats = relay.reshape(adj_feats, [n_faces, 3, in_feats])
    for p in range(3):
        p_feats = relay.strided_slice(adj_feats, relay.const([0, p], itype),
                                      relay.const([n_faces, p + 1], itype))
        pp1_feats = relay.strided_slice(
            adj_feats, relay.const([0, (p + 1) % 3], itype),
            relay.const([n_faces, (p + 1) % 3 + 1], itype))
        p_feats = relay.reshape(p_feats, [n_faces, in_feats])
        pp1_feats = relay.reshape(pp1_feats, [n_faces, in_feats])
        sum1 = relay.add(sum1, p_feats)
        sum2 = relay.add(sum2, relay.abs(relay.subtract(p_feats, pp1_feats)))
        sum3 = relay.add(sum3, relay.abs(relay.subtract(p_feats, x)))
    return relay.nn.matmul(x, w0) + relay.nn.matmul(sum1, w1) + relay.nn.matmul(
        sum2, w2) + relay.nn.matmul(sum3, w3)


y = get_subdivnet_relay()
args = relay.analysis.free_vars(y)
net = relay.Function(args, y)
mod = tvm.IRModule.from_expr(net)
mod = relay.transform.InferType()(mod)
# print(mod.astext(show_meta_data=False))

#####################################################################
# Run the generate library

if target_name.startswith('llvm'):
    dev = tvm.cpu()
elif target_name.startswith('cuda'):
    dev = tvm.cuda()
else:
    assert False

params = {"w" + str(p): tvm.nd.array(d_w[p]) for p in range(4)}

# # Module without tuning
# d_adj = np.zeros((n_faces, 3)).astype(itype)
# d_x = np.random.uniform(-1, 1, size=(n_faces, in_feats)).astype(dtype)
# d_w = np.random.uniform(-1, 1, size=(3, in_feats, out_feats)).astype(dtype)
#
#opt_level = 3
#with tvm.transform.PassContext(opt_level=opt_level):
#    lib = relay.build(mod, target, params=params)
#
# module = graph_executor.GraphModule(lib["default"](dev))

################################################################################
# Tune the model

## AutoTVM (unable to tune)
#
#number = 10
#repeat = 1
#min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
#timeout = 10  # in seconds
#
## create a TVM runner
#runner = autotvm.LocalRunner(
#    number=number,
#    repeat=repeat,
#    timeout=timeout,
#    min_repeat_ms=min_repeat_ms,
#    enable_cpu_cache_flush=True,
#)
#
#tuning_option = {
#    "tuner":
#        "xgb",
#    "trials":
#        10,
#    "early_stopping":
#        100,
#    "measure_option":
#        autotvm.measure_option(
#            builder=autotvm.LocalBuilder(build_func="default"),
#            runner=runner),
#}
#
## tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
#tasks = autotvm.task.extract_from_program(mod["main"],
#                                          target=target,
#                                          params=params)
#print(len(tasks))
#print(tasks)
#
## Tune the extracted tasks sequentially.
#for i, task in enumerate(tasks):
#    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
#    tuner_obj = XGBTuner(task, loss_type="rank")
#    tuner_obj.tune(
#        n_trial=min(tuning_option["trials"], len(task.config_space)),
#        early_stopping=tuning_option["early_stopping"],
#        measure_option=tuning_option["measure_option"],
#        callbacks=[
#            autotvm.callback.progress_bar(tuning_option["trials"],
#                                          prefix=prefix),
#            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
#        ],
#    )
#
#with autotvm.apply_history_best(tuning_option["tuning_records"]):
#    with tvm.transform.PassContext(opt_level=3, config={}):
#        lib = relay.build(mod, target=target, params=params)
#        # lib = relay.build(mod, target=target)
#
#dev = tvm.device(str(target), 0)
#module = graph_executor.GraphModule(lib["default"](dev))

# AutoSchedule

if cmd_args.is_tuning:

    print()
    print("!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!! TUNING             !!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!")
    print()

    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"], params, target, include_simple_tasks=True)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    print('#Tuning trials', cmd_args.tuning_rounds * len(tasks))
    print(task_weights)
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
print("!! EVALUATING         !!")
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print()

with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)
module = graph_executor.GraphModule(lib["default"](dev))

################################################################################
# Comparing the Tuned and Untuned Models

# create module
module.set_input("adj", d_adj)
module.set_input("x", d_x)
module.run()
y = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
store_txt("y.out", y)

import timeit

print(
    f"{cmd_args.warmup_num} warmup, {cmd_args.test_num} repeats for evalution")
timing_number = 1
timeit.Timer(lambda: module.run()).repeat(repeat=cmd_args.warmup_num, number=1)
if cmd_args.profile_gpu:
    profile_start()
optimized = (np.array(
    timeit.Timer(lambda: module.run()).repeat(
        repeat=cmd_args.test_num, number=timing_number)) * 1000 / timing_number)
if cmd_args.profile_gpu:
    profile_stop()
optimized = {
    "mean": np.mean(optimized),
    "median": np.median(optimized),
    "std": np.std(optimized)
}

print("Time (ms): %s" % (optimized))
