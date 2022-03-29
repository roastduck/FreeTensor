# FAILING LOG:
#
# No valid state found in this search round. Check if it has traversed all of the search space.
# /home/zhenly/.local/lib/python3.8/site-packages/xgboost/compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
#   from pandas import MultiIndex, Int64Index
# /home/zhenly/.local/lib/python3.8/site-packages/xgboost/training.py:17: UserWarning: Old style callback is deprecated.  See: https://xgboost.readthedocs.io/en/latest/python/callbacks.html
#   warnings.warn(f'Old style callback is deprecated.  See: {link}', UserWarning)
# Traceback (most recent call last):
#   File "./tvm_compute_fix.py", line 150, in <module>
#     sch, args = task.apply_best(log_file)
#   File "/home/zhenly/App/tvm-211104/python/tvm/auto_scheduler/search_task.py", line 521, in apply_best
#     raise RuntimeError(
# RuntimeError: Cannot find any valid schedule for ["compute_sum_on_1dim", 1024, 13, 64, "float32"] in file matmul.json

import os

import numpy as np
import tvm
from tvm import te, auto_scheduler
from tvm import topi

target_name = 'llvm'
target = tvm.target.Target(target_name)
# TODO: set correct shapes
n_faces, in_feats, out_feats = 1024, 13, 64

################################################################################
# Default schedule

itype = 'int32'

# Default schedule
if True:

    def get_subdivnet_compute(n_faces, in_feats, out_feats, dtype):
        adj = te.placeholder((n_faces, 3), name="adj", dtype=itype)
        x = te.placeholder((n_faces, in_feats), name="x", dtype=dtype)
        w1 = te.placeholder((in_feats, out_feats), name="w1", dtype=dtype)

        # k = te.reduce_axis((0, in_feats), name="k")
        p = te.reduce_axis((0, 3), name="p")
        # There are three simplified te.compute codes with indirect mem access
        sum1 = te.compute(
            (n_faces,),
            lambda i: te.sum(x[adj[i, p], 0], axis=p),
            name="Kernel-w1",
        )
        return [adj, x, w1, sum1]

    A1, A2, A3, C = get_subdivnet_compute(n_faces, in_feats, out_feats,
                                          'float32')
    s = te.create_schedule(C.op)
    fadd = tvm.build(s, [A1, A2, A3, C], target, name="myadd")
    dev = tvm.device(target.kind.name, 0)
    print(tvm.lower(s, [A1, A2, A3, C], simple_mode=True))

    ###########################################################################

    func = tvm.build(s, [A1, A2, A3, C], target)
    adj_np = np.random.uniform(size=(n_faces, 3)).astype(np.int32)
    x_np = np.random.uniform(size=(n_faces, in_feats)).astype(np.float32)
    w1_np = np.random.uniform(size=(in_feats, out_feats)).astype(np.float32)
    idx_np = np.asarray((0, 1)).astype(np.int32)
    idx2_np = np.asarray((-1,)).astype(np.int32)

    if target_name.startswith('llvm'):
        dev = tvm.cpu()
    elif target_name == 'cuda':
        dev = tvm.cuda()
    else:
        assert False
    adj_tvm = tvm.nd.array(adj_np, device=dev)
    x_tvm = tvm.nd.array(x_np, device=dev)
    w1_tvm = tvm.nd.array(w1_np, device=dev)
    idx_tvm = tvm.nd.array(idx_np, device=dev)
    idx2_tvm = tvm.nd.array(idx2_np, device=dev)
    # out_tvm = tvm.nd.empty((2,3), device=dev)
    out_tvm = tvm.nd.empty((n_faces,), device=dev)
    print(adj_np, idx_np.shape, idx2_np.shape)
    # func(adj_tvm, x_tvm, idx_tvm, out_tvm)
    func(adj_tvm, x_tvm, w1_tvm, out_tvm)
    print(out_tvm)

# Ansor
if True:

    @auto_scheduler.register_workload
    def compute_sum_on_1dim(n_faces, in_feats, out_feats, dtype):
        adj = te.placeholder((n_faces, 3), name="adj", dtype=itype)
        x = te.placeholder((n_faces, in_feats), name="x", dtype=dtype)
        w1 = te.placeholder((in_feats, out_feats), name="w1", dtype=dtype)

        p = te.reduce_axis((0, 3), name="p")
        sum1 = te.compute(
            (n_faces,),
            lambda i: te.sum(x[adj[i, p], 0], axis=p),
            name="Kernel-w1",
        )
        return [adj, x, w1, sum1]

    @auto_scheduler.register_workload
    def compute_complete(n_faces, in_feats, out_feats, dtype):
        adj = te.placeholder((n_faces, 3), name="adj", dtype=itype)
        x = te.placeholder((n_faces, in_feats), name="x", dtype=dtype)
        w1 = te.placeholder((in_feats, out_feats), name="w1", dtype=dtype)

        k = te.reduce_axis((0, in_feats), name="k")
        p = te.reduce_axis((0, 3), name="p")
        sum1 = te.compute(
            (in_feats,),
            lambda i: te.sum(x[adj[i, p], k], axis=(p, k)),
            name="Kernel-w1",
            # enable automatic layout transform for tensor B
            # attrs={"layout_free_placeholders": [B]},
        )
        # out = te.compute((N, M), lambda i, j: matmul[i, j] + C[i, j], name="out")
        return [adj, x, w1, sum1]

    ################################################################################
    # .. note:: Improve performance with custom targets
    #   In order for TVM to take full advantage of specific hardware platforms,
    #   you will want to manuall specify your CPU capabilities. For example:
    #   - replace "llvm" below with "llvm -mcpu=core-avx2" to enable AVX2
    #   - replace "llvm" below with "llvm -mcpu=skylake-avx512" to enable AVX-512
    task = tvm.auto_scheduler.SearchTask(func=compute_sum_on_1dim,
                                         args=(n_faces, in_feats, out_feats,
                                               "float32"),
                                         target=target)

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    ################################################################################
    # Set Parameters for Auto-Scheduler
    # ---------------------------------
    # Next, we set parameters for the auto-scheduler.
    #
    # * :code:`num_measure_trials` is the number of measurement trials we can use
    #   during the search.  We only make 10 trials in this tutorial for a fast
    #   demonstration. In practice, 1000 is a good value for the search to converge.
    #   You can do more trials according to your time budget.
    # * In addition, we use :code:`RecordToFile` to log measurement records into a
    #   file `matmul.json`.  The measurement records can be used to query the history
    #   best, resume the search, and do more analyses later.
    # * see :any:`auto_scheduler.TuningOptions` for more parameters

    log_file = "matmul.json"
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=7,
    )

    ################################################################################
    # Run the search
    # --------------
    # Now we get all inputs ready. Pretty simple, isn't it?  We can kick off the
    # search and let the auto-scheduler do its magic.  After some measurement
    # trials, we can load the best schedule from the log file and apply it.

    # Run auto-tuning (search)
    task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_file)

    ################################################################################
    # Inspecting the Optimized Schedule
    # ---------------------------------
    # We can lower the schedule to see the IR after auto-scheduling.  The
    # auto-scheduler correctly performs optimizations including multi-level tiling,
    # layout transformation, parallelization, vectorization, unrolling, and
    # operator fusion.

    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))

    exit()

    ################################################################################
    # Check correctness and evaluate performance
    # ------------------------------------------
    # We build the binary and check its correctness and performance.

    func = tvm.build(sch, args, target)
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = np.random.uniform(size=(N, M)).astype(np.float32)
    out_np = a_np.dot(b_np) + c_np

    if target_name.startswith('llvm'):
        dev = tvm.cpu()
    elif target_name == 'cuda':
        dev = tvm.cuda()
    else:
        assert False
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, device=dev)
    func(a_tvm, b_tvm, c_tvm, out_tvm)

    # Check results
    np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    print("Execution time of this operator: %.3f ms" %
          (np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results) * 1000))
