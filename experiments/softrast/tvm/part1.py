import timeit
import numpy as np

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
parser.add_argument('--tune',
                    action='store_true',
                    dest='is_tuning',
                    help='(this argument is ignored in this part)')
parser.add_argument('--tune-rounds',
                    type=int,
                    default=1000,
                    dest='tuning_rounds',
                    help='(this argument is ignored in this part)')
parser.add_argument('--eval',
                    default='',
                    dest='eval',
                    help='(this argument is ignored in this part)')
parser.add_argument('--warmup-repeat', type=int, default=10, dest='warmup_num')
parser.add_argument('--timing-repeat', type=int, default=100, dest='test_num')
cmd_args = parser.parse_args()

if cmd_args.target == 'cpu':
    target_name = 'llvm -libs=mkl -mcpu=core-avx2'
    dev = tvm.cpu()
elif cmd_args.target == 'gpu':
    target_name = 'cuda -libs=cublas'
    dev = tvm.cuda()
else:
    assert False

tuning_rounds = 1000
target = tvm.target.Target(target_name)
dtype, itype = 'float32', 'int32'
time_now = datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
log_file = f'ansor.{sys.argv[1]}.{time_now}.json'

vertices = load_txt("../vertices.in", "float32")
faces = load_txt("../faces.in", "int32")
n_verts = vertices.shape[0]
n_faces = faces.shape[0]
h = 64
w = 64
y = np.zeros((n_faces, 3, 3), dtype="float32")
output_shape = (n_faces, 3, 3)


def get_congtiguous_relay(n_verts,
                          n_faces,
                          h,
                          w,
                          sigma=1e-4,
                          dtype='float32',
                          itype='int32'):
    vertices = relay.var("vertices", shape=(n_verts, 3), dtype=dtype)
    faces = relay.var("faces", shape=(n_faces, 3), dtype=itype)
    v = relay.adv_index([vertices, faces])
    return v


y = get_congtiguous_relay(n_verts, n_faces, h, w)
args = relay.analysis.free_vars(y)
net = relay.Function(args, y)
mod = tvm.IRModule.from_expr(net)
mod = relay.transform.InferType()(mod)
print(mod.astext(show_meta_data=False))

######################################################################
# Compilation

# d_adj = np.zeros((n_faces, 3)).astype(itype)
# d_x = np.random.uniform(-1, 1, size=(n_faces, in_feats)).astype(dtype)
# d_w = np.random.uniform(-1, 1, size=(3, in_feats, out_feats)).astype(dtype)

opt_level = 3
params = {"vertices": tvm.nd.array(vertices), "faces": tvm.nd.array(faces)}

#####################################################################
# # Module without tuning
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)
    print('Successfully built without tuning.')
    # lib = relay.build(mod, target, params={})

# Run the generate library
module = graph_executor.GraphModule(lib["default"](dev))

################################################################################
# We failed to tune the model. Failed approaches are as below:

# Failed approach 1: AutoTVM

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
#        tuning_rounds,
#    "early_stopping":
#        100,
#    "measure_option":
#        autotvm.measure_option(
#            builder=autotvm.LocalBuilder(build_func="default"),
#            runner=runner),
#    "tuning_records":
#        log_file,
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
#module = graph_executor.GraphModule(lib["default"](dev))

# Failed approach 2: AutoSchedule
#tasks, task_weights = auto_scheduler.extract_tasks(
#    mod["main"], params, target, include_simple_tasks=True)
#tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
#print('#Tuning trials', tuning_rounds * len(tasks))
#print(task_weights)
#print([task.compute_dag for task in tasks])
#tune_option = auto_scheduler.TuningOptions(
#    num_measure_trials=tuning_rounds * len(tasks),
#    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
#    verbose=2,
#)
#tuner.tune(tune_option)
#with auto_scheduler.ApplyHistoryBest(log_file):
#    with tvm.transform.PassContext(
#            opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
#        lib = relay.build(mod, target=target, params=params)
#module = graph_executor.GraphModule(lib["default"](dev))

print()
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print("!! EVALUATING PART 1  !!")
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print()

# create module
# module.set_input("", tvm.nd.array(feat_np))
module.run()
y = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
store_txt("v.tmp", y)

print(
    f"{cmd_args.warmup_num} warmup, {cmd_args.test_num} repeats for evalution")
timing_number = 1
timeit.Timer(lambda: module.run()).repeat(repeat=cmd_args.warmup_num, number=1)
optimized = (np.array(
    timeit.Timer(lambda: module.run()).repeat(
        repeat=cmd_args.test_num, number=timing_number)) * 1000 / timing_number)
optimized = {
    "mean": np.mean(optimized),
    "median": np.median(optimized),
    "std": np.std(optimized)
}

print("optimized: %s" % (optimized))
