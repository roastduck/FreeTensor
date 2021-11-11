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
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

######################################################################
# Define Neural Network in Relay

# target_name = 'llvm -libs=mkl'
target_name = 'cuda -libs=cublas'
target = tvm.target.Target(target_name)
dtype, itype = 'float32', 'int32'

d_adj = np.loadtxt("../adj.in", dtype=np.int32)
n_faces, in_feats, out_feats = d_adj.shape[0], 13, 64
d_x = np.loadtxt("../x.in", dtype=dtype)
d_w = [np.loadtxt(f"../w{i}.in", dtype=dtype) for i in range(4)]
d_y = np.loadtxt("../d_y.in", dtype=dtype)
output_shape=(n_faces, out_feats)


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
        p_feats = relay.strided_slice(adj_feats, relay.const(
            [0, p], itype), relay.const([n_faces, p+1], itype))
        pp1_feats = relay.strided_slice(adj_feats, relay.const(
            [0, (p+1)%3], itype), relay.const([n_faces, (p+1)%3+1], itype))
        p_feats = relay.reshape(p_feats, [n_faces, in_feats])
        pp1_feats = relay.reshape(pp1_feats, [n_faces, in_feats])
        sum1 = relay.add(sum1, p_feats)
        sum2 = relay.add(sum2, relay.abs(relay.subtract(p_feats, pp1_feats)))
        sum3 = relay.add(sum3, relay.abs(relay.subtract(p_feats, x)))
    return relay.nn.matmul(x, w0)+relay.nn.matmul(sum1, w1) +relay.nn.matmul(sum2, w2)+relay.nn.matmul(sum3, w3)

y = get_subdivnet_relay()
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
params = {"w"+str(p): tvm.nd.array(d_w[p]) for p in range(4)}
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target,params=params)

#####################################################################
# Run the generate library

if target_name.startswith('llvm'):
    dev = tvm.cpu()
elif target_name.startswith('cuda'):
    dev = tvm.cuda()
else:
    assert False

# # Module without tuning 
# module = graph_executor.GraphModule(lib["default"](dev))

################################################################################
# Tune the model

if True:
    number = 10
    repeat = 1
    min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
    timeout = 10  # in seconds

    # create a TVM runner
    runner = autotvm.LocalRunner(
        number=number,
        repeat=repeat,
        timeout=timeout,
        min_repeat_ms=min_repeat_ms,
        enable_cpu_cache_flush=True,
    )

    tuning_option = {
        "tuner": "xgb",
        "trials": 10,
        "early_stopping": 100,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="default"), runner=runner
        ),
        "tuning_records": "resnet-50-v2-autotuning.json",
    }

    # tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
    print(len(tasks))
    print(tasks)

    # Tune the extracted tasks sequentially.
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = XGBTuner(task, loss_type="rank")
        tuner_obj.tune(
            n_trial=min(tuning_option["trials"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )

    with autotvm.apply_history_best(tuning_option["tuning_records"]):
        with tvm.transform.PassContext(opt_level=3, config={}):
            lib = relay.build(mod, target=target, params=params)
            # lib = relay.build(mod, target=target)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))

# buggy Ansor
if False:
    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"], params, target, include_simple_tasks=True)
    print(tasks)
    log_file='ansor.json'
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(
        number=5, repeat=1, min_repeat_ms=1000, timeout=120)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(num_measure_trials=2000,
                                               runner=measure_ctx.runner,
                                               measure_callbacks=[
                                                   auto_scheduler.RecordToFile(log_file)],)
    tuner.tune(tune_option)
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
    module = graph_executor.GraphModule(lib["default"](dev))


################################################################################
# Comparing the Tuned and Untuned Models

# create module
module.set_input("adj", d_adj)
module.set_input("x", d_x)
module.run()
y = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
np.savetxt("y.out", y)

import timeit
warmup_num = 10
timing_number = 1
timing_repeat = 1000
timeit.Timer(lambda: module.run()).repeat(repeat=warmup_num, number=1)
optimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(
        repeat=timing_repeat, number=timing_number))
    * 1000 / timing_number
)
optimized = {"mean": np.mean(optimized), "median": np.median(
    optimized), "std": np.std(optimized)}


print("optimized: %s" % (optimized))