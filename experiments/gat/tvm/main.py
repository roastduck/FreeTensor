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
from datetime import datetime
import logging
# Enable debug logs
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

sys.path.append('../..')
from common.numpy.io import load_txt, store_txt

######################################################################
# Define Neural Network in Relay

if len(sys.argv) != 3:
    print(len(sys.argv))
    print(f"Usage: {sys.argv[0]} <cpu/gpu>")
    exit(-1)
if sys.argv[1] == 'cpu':
    target_name = 'llvm -libs=mkl -mcpu=core-avx2'
    dev = tvm.cpu()
elif sys.argv[1] == 'gpu':
    target_name = 'cuda -libs=cublas'
    dev = tvm.cuda()
else:
    assert False

tuning_rounds = 1000
target = tvm.target.Target(target_name)
dtype, itype = 'float32', 'int32'
time_now = datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
log_file = f'ansor.{sys.argv[1]}.{time_now}.json'

ptr_np = load_txt("../ptr.in", itype)
idx_np = load_txt("../idx.in", itype)
# ptr_np = np.loadtxt("../ptr.in", dtype=itype)
# idx_np = np.loadtxt("../idx.in", dtype=itype)
feat_np = load_txt("../x.in", dtype)
# feat_np = np.loadtxt("../x.in", dtype=dtype)
weight_np = load_txt("../w.in", dtype)
# weight_np = np.loadtxt("../w.in", dtype=dtype)
attn_l_np = np.expand_dims(load_txt("../w_attn_1.in", dtype), axis=1)
attn_r_np = np.expand_dims(load_txt("../w_attn_2.in", dtype), axis=1)
# attn_l_np = np.expand_dims(np.loadtxt("../w_attn_1.in", dtype=dtype), axis=1)
# attn_r_np = np.expand_dims(np.loadtxt("../w_attn_2.in", dtype=dtype), axis=1)
d_y_np = load_txt("../d_y.in", dtype)
# d_y_np = np.loadtxt("../d_y.in", dtype=dtype)
num_v = ptr_np.shape[0] - 1
num_e = idx_np.shape[0]
idx_center_np = np.zeros([num_e], dtype=itype)
for i in range(num_v):
    idx_center_np[ptr_np[i]:ptr_np[i + 1]] = i
print(feat_np.shape, weight_np.shape)

feat_len = 32

output_shape = (num_v, feat_len)


def get_spmm_relay(num_v, num_e, feat_len, dtype, itype):
    ptr = relay.var("ptr", shape=(num_v + 1,), dtype=itype)
    idx = relay.var("idx", shape=(num_e,), dtype=itype)
    idx_center = relay.var("idx_center", shape=(num_e,), dtype=itype)
    feat = relay.var("feat", shape=(num_v, feat_len), dtype=dtype)
    weight = relay.var("weight", shape=(feat_len, feat_len), dtype=dtype)
    attn_l = relay.var("attn_l", shape=(feat_len, 1), dtype=dtype)
    attn_r = relay.var("attn_r", shape=(feat_len, 1), dtype=dtype)

    edge_exp_oe = relay.squeeze(relay.exp(relay.ones([num_e, 1], dtype=dtype)),
                                axis=[1])
    edge_sum_on = relay.nn.sparse_dense(relay.ones([1, num_v], dtype=dtype),
                                        [ptr, idx, edge_exp_oe],
                                        sparse_lhs=True)
    return edge_sum_on


def get_gat_relay(num_v, num_e, feat_len, dtype, itype):
    ptr = relay.var("ptr", shape=(num_v + 1,), dtype=itype)
    idx = relay.var("idx", shape=(num_e,), dtype=itype)
    idx_center = relay.var("idx_center", shape=(num_e,), dtype=itype)
    feat = relay.var("feat", shape=(num_v, feat_len), dtype=dtype)
    weight = relay.var("weight", shape=(feat_len, feat_len), dtype=dtype)
    attn_l = relay.var("attn_l", shape=(feat_len, 1), dtype=dtype)
    attn_r = relay.var("attn_r", shape=(feat_len, 1), dtype=dtype)

    feat2 = relay.nn.matmul(feat, weight)
    att_l = relay.nn.matmul(feat2, attn_l)
    att_r = relay.nn.matmul(feat2, attn_r)
    att_l_oe = relay.adv_index([att_l, idx_center])
    att_r_oe = relay.adv_index([att_r, idx_center])
    edge_exp_oe = relay.squeeze(relay.exp(
        relay.nn.leaky_relu(att_l_oe + att_r_oe)),
                                axis=[1])
    edge_sum_on = relay.nn.sparse_dense(relay.ones([1, num_v], dtype=dtype),
                                        [ptr, idx, edge_exp_oe],
                                        sparse_lhs=True)
    yy = relay.nn.sparse_dense(relay.transpose(feat2), [ptr, idx, edge_exp_oe],
                               sparse_lhs=True)
    y = yy / edge_sum_on
    return y


y = get_gat_relay(num_v, num_e, feat_len, dtype, itype)
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
params = {
    "ptr": tvm.nd.array(ptr_np),
    "idx": tvm.nd.array(idx_np),
    "idx": tvm.nd.array(idx_np),
    "idx_center": tvm.nd.array(idx_center_np),
    "weight": tvm.nd.array(weight_np),
    "attn_l": tvm.nd.array(attn_l_np),
    "attn_r": tvm.nd.array(attn_r_np)
}
for k, v in params.items():
    print(k, v.shape)

#####################################################################
# # Module without tuning
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)
    print('Successfully built without tuning.')
    # lib = relay.build(mod, target, params={})

# Run the generate library
module = graph_executor.GraphModule(lib["default"](dev))

################################################################################
# Tune the model

if False:
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
        "tuner":
            "xgb",
        "trials":
            tuning_rounds,
        "early_stopping":
            100,
        "measure_option":
            autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func="default"),
                runner=runner),
        "tuning_records":
            log_file,
    }

    # tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
    tasks = autotvm.task.extract_from_program(mod["main"],
                                              target=target,
                                              params=params)
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
                autotvm.callback.progress_bar(tuning_option["trials"],
                                              prefix=prefix),
                autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )

    with autotvm.apply_history_best(tuning_option["tuning_records"]):
        with tvm.transform.PassContext(opt_level=3, config={}):
            lib = relay.build(mod, target=target, params=params)
            # lib = relay.build(mod, target=target)

    module = graph_executor.GraphModule(lib["default"](dev))

if True:
    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"], params, target, include_simple_tasks=True)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    print('#Tuning trials', tuning_rounds * len(tasks))
    print(task_weights)
    print([task.compute_dag for task in tasks])
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=tuning_rounds * len(tasks),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    tuner.tune(tune_option)
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
    module = graph_executor.GraphModule(lib["default"](dev))

################################################################################
# Comparing the Tuned and Untuned Models

# create module
module.set_input("feat", tvm.nd.array(feat_np))
module.run()
y = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
np.savetxt("y.out", y)

warmup_num = 10
timing_number = 1
timing_repeat = 1000
timeit.Timer(lambda: module.run()).repeat(repeat=warmup_num, number=1)
optimized = (np.array(
    timeit.Timer(lambda: module.run()).repeat(
        repeat=timing_repeat, number=timing_number)) * 1000 / timing_number)
optimized = {
    "mean": np.mean(optimized),
    "median": np.median(optimized),
    "std": np.std(optimized)
}

print("optimized: %s" % (optimized))
