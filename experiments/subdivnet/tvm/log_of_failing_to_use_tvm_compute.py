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


def get_subdivnet_compute(n_faces, in_feats, out_feats, dtype):
    adj = te.placeholder((n_faces, 3), name="adj", dtype=dtype)
    x = te.placeholder((n_faces, in_feats), name="x", dtype=dtype)
    w1 = te.placeholder((in_feats, out_feats), name="w1", dtype=dtype)

    # k = te.reduce_axis((0, in_feats), name="k")
    p = te.reduce_axis((0, 3), name="p")
    # There are three simplified te.compute codes with indirect mem access
    sum1 = te.compute(
        (in_feats,),
        lambda i: te.sum(x[adj[i, p], 0], axis=p),
        name="Kernel-w1",
    )
    # # Error log
    # Computational DAG:
    # adj = PLACEHOLDER [1024, 3]
    # x = PLACEHOLDER [1024, 13]
    # Kernel-w1(i) += x[adj[i, p], k]

    # Traceback (most recent call last):
    #   File "./main.py", line 131, in <module>
    #     fadd = tvm.build(s, [A1, A2, A3, C], target, name="myadd")
    #   File "/home/zhenly/App/tvm-211104/python/tvm/driver/build_module.py", line 263, in build
    #     rt_mod_host = _driver_ffi.preprocess_module(target_input_mod, target_host)
    #   File "/home/zhenly/App/tvm-211104/python/tvm/_ffi/_ctypes/packed_func.py", line 237, in __call__
    #     raise get_last_ffi_error()
    # tvm._ffi.base.TVMError: Traceback (most recent call last):
    #   5: TVMFuncCall
    #   4: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::runtime::Module (tvm::runtime::Map<tvm::Target, tvm::IRModule, void, void> const&, tvm::Target)>::Assign
    # TypedLambda<tvm::{lambda(tvm::runtime::Map<tvm::Target, tvm::IRModule, void, void> const&, tvm::Target)#6}>(tvm::{lambda(tvm::runtime::Map<tvm::Target, tvm::IRModule, void, void> const&, tvm::Target)#6}, std::__cxx11::basic_s
    # tring<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
    #   3: tvm::PreProcessModuleForBuild(tvm::runtime::Map<tvm::Target, tvm::IRModule, void, void> const&, tvm::Target const&)
    #   2: tvm::codegen::Build(tvm::IRModule, tvm::Target)
    #   1: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::runtime::Module (tvm::IRModule, tvm::Target)>::AssignTypedLambda<tvm::codegen::{lambda(tvm::IRModule, tv
    # m::Target)#1}>(tvm::codegen::{lambda(tvm::IRModule, tvm::Target)#1}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_
    # invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
    #   0: tvm::codegen::LLVMModuleNode::Init(tvm::IRModule const&, tvm::Target const&)
    #   File "/home/zhenly/App/tvm-211104/src/target/llvm/llvm_module.cc", line 327
    # TVMError: LLVM module verification failed with the following errors:
    # SExt only operates on integer
    #   %8 = sext float %7 to i64
    # SExt only operates on integer
    #   %16 = sext float %15 to i64
    # SExt only operates on integer
    #   %24 = sext float %23 to i64
    # SExt only operates on integer
    #   %34 = sext float %33 to i64
    sum1 = te.compute(
        (in_feats,),
        lambda i: te.sum(x[adj[i, 0], p], axis=p),
        name="Kernel-w1",
    )
    # # Error log
    # Computational DAG:
    # adj = PLACEHOLDER [1024, 3]
    # x = PLACEHOLDER [1024, 13]
    # Kernel-w1(i) += x[adj[i, p], k]

    # Traceback (most recent call last):
    #   File "./main.py", line 166, in <module>
    #     fadd = tvm.build(s, [A1, A2, A3, C], target, name="myadd")
    #   File "/home/zhenly/App/tvm-211104/python/tvm/driver/build_module.py", line 263, in build
    #     rt_mod_host = _driver_ffi.preprocess_module(target_input_mod, target_host)
    #   File "/home/zhenly/App/tvm-211104/python/tvm/_ffi/_ctypes/packed_func.py", line 237, in __call__
    #     raise get_last_ffi_error()
    # tvm._ffi.base.TVMError: Traceback (most recent call last):
    #   5: TVMFuncCall
    #   4: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::runtime::Module (tvm::runtime::Map<tvm::Target, tvm::IRModule, void, void> const&, tvm::Target)>::AssignTypedLambda<tvm::{lambda(tvm::runtime::Map<tvm::Target, tvm::IRModule, void, void> const&, tvm::Target)#6}>(tvm::{lambda(tvm::runtime::Map<tvm::Target, tvm::IRModule, void, void> const&, tvm::Target)#6}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
    #   3: tvm::PreProcessModuleForBuild(tvm::runtime::Map<tvm::Target, tvm::IRModule, void, void> const&, tvm::Target const&)
    #   2: tvm::codegen::Build(tvm::IRModule, tvm::Target)
    #   1: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::runtime::Module (tvm::IRModule, tvm::Target)>::AssignTypedLambda<tvm::codegen::{lambda(tvm::IRModule, tvm::Target)#1}>(tvm::codegen::{lambda(tvm::IRModule, tvm::Target)#1}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
    #   0: tvm::codegen::LLVMModuleNode::Init(tvm::IRModule const&, tvm::Target const&)
    #   File "/home/zhenly/App/tvm-211104/src/target/llvm/llvm_module.cc", line 327
    # TVMError: LLVM module verification failed with the following errors:
    # SExt only operates on integer
    #   %279 = sext float %278 to i64
    # SExt only operates on integer
    #   %284 = sext float %283 to i64
    # SExt only operates on integer
    #   %289 = sext float %288 to i64
    sum1 = te.compute(
        (in_feats,),
        lambda i: te.sum(x[adj[0, 0], p], axis=p),
        name="Kernel-w1",
    )
    # # Error log
    # Computational DAG:
    # adj = PLACEHOLDER [1024, 3]
    # x = PLACEHOLDER [1024, 13]
    # Kernel-w1(i) += x[adj[i, p], k]

    # Traceback (most recent call last):
    #   File "./main.py", line 193, in <module>
    #     fadd = tvm.build(s, [A1, A2, A3, C], target, name="myadd")
    #   File "/home/zhenly/App/tvm-211104/python/tvm/driver/build_module.py", line 263, in build
    #     rt_mod_host = _driver_ffi.preprocess_module(target_input_mod, target_host)
    #   File "/home/zhenly/App/tvm-211104/python/tvm/_ffi/_ctypes/packed_func.py", line 237, in __call__
    #     raise get_last_ffi_error()
    # tvm._ffi.base.TVMError: Traceback (most recent call last):
    #   5: TVMFuncCall
    #   4: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::runtime::Module (tvm::runtime::Map<tvm::Target, tvm::IRModule, void, void> const&, tvm::Target)>::AssignTypedLambda<tvm::{lambda(tvm::runtime::Map<tvm::Target, tvm::IRModule, void, void> const&, tvm::Target)#6}>(tvm::{lambda(tvm::runtime::Map<tvm::Target, tvm::IRModule, void, void> const&, tvm::Target)#6}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
    #   3: tvm::PreProcessModuleForBuild(tvm::runtime::Map<tvm::Target, tvm::IRModule, void, void> const&, tvm::Target const&)
    #   2: tvm::codegen::Build(tvm::IRModule, tvm::Target)
    #   1: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::runtime::Module (tvm::IRModule, tvm::Target)>::AssignTypedLambda<tvm::codegen::{lambda(tvm::IRModule, tvm::Target)#1}>(tvm::codegen::{lambda(tvm::IRModule, tvm::Target)#1}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
    #   0: tvm::codegen::LLVMModuleNode::Init(tvm::IRModule const&, tvm::Target const&)
    #   File "/home/zhenly/App/tvm-211104/src/target/llvm/llvm_module.cc", line 327
    # TVMError: LLVM module verification failed with the following errors:
    # SExt only operates on integer
    #   %7 = sext float %6 to i64
    # SExt only operates on integer
    #   %12 = sext float %11 to i64
    # SExt only operates on integer
    #   %17 = sext float %16 to i64
    return [adj, x, w1, sum1]


# A1, A2, A3, C = get_subdivnet_compute(n_faces, in_feats, out_feats, 'float32')
# s = te.create_schedule(C.op)
# fadd = tvm.build(s, [A1, A2, A3, C], target, name="myadd")
# dev = tvm.device(target.kind.name, 0)
# print(tvm.lower(s, [A1, A2, A3, C], simple_mode=True))


def get_test(n_faces, in_feats, out_feats, dtype='float32', itype='int32'):
    adj = te.placeholder((n_faces, 3), name="adj", dtype=dtype)
    x = te.placeholder((n_faces, in_feats), name="x", dtype=dtype)
    # w1 = te.placeholder((in_feats, out_feats), name="w1", dtype=dtype)
    idx = te.placeholder((2,), name="idx", dtype=itype)
    # adj_flatten = topi.reshape(adj, [n_faces, 3])
    # adj_feats = topi.adv_index(x, [adj_flatten]).reshape([n_faces, 3, in_feats])
    # for p in range(3):
    #     adj_feats = topi.adv_index(adj_feats, [[], [p]])
    adj_feats = topi.strided_slice(adj, topi.const_vector([0, 1]),
                                   topi.const_vector([n_faces, 2]))

    return [adj, x, idx, adj_feats]


# n_faces, in_feats, out_feats,
def get_subdivnet_topi(dtype='float32', itype='int32'):
    adj = te.placeholder((n_faces, 3), name="adj", dtype=itype)
    x = te.placeholder((n_faces, in_feats), name="x", dtype=dtype)
    w1 = te.placeholder((in_feats, out_feats), name="w1", dtype=dtype)
    sum1 = te.placeholder((n_faces, in_feats), name="sum1", dtype=dtype)
    idx = te.placeholder((2,), name="idx", dtype=itype)

    adj_flatten = topi.reshape(adj, [n_faces, 3])
    adj_feats = topi.adv_index(x, [adj_flatten])
    adj_feats = topi.reshape(adj_feats, [n_faces, 3, in_feats])
    print(sum1.shape)
    # for p in range(3):
    # p_feats = topi.strided_slice(adj_feats, topi.const_vector(
    #     [0, p]), topi.const_vector([n_faces, p+1]))
    # p_feats = topi.reshape(p_feats, [n_faces, in_feats])
    # print(sum1.shape, adj_feats, p_feats)
    # sum1 = sum1+p_feats
    p = 0
    p_feats = topi.strided_slice(adj_feats, topi.const_vector([0, p]),
                                 topi.const_vector([n_faces, p + 1]))
    p_feats = topi.reshape(p_feats, [n_faces, in_feats])
    print(sum1.shape, adj_feats, p_feats)
    sum1 = sum1 + p_feats

    new_feats = topi.matmul(sum1, w1)

    return [adj, x, idx, new_feats]


# args = get_subdivnet_topi(n_faces, in_feats, out_feats)
args = get_subdivnet_topi()
# sch = topi.x86.schedule_elemwise(args[-1])
sch = topi.x86.schedule_batch_matmul(args[-1])
print(tvm.lower(sch, args, simple_mode=True))


def get_subdivnet_compute(n_faces,
                          in_feats,
                          out_feats,
                          dtype='float32',
                          itype='int32'):
    w1 = te.placeholder((in_feats, out_feats), name="w1", dtype=dtype)


###########################################################################

func = tvm.build(sch, args, target)
adj_np = np.random.uniform(size=(n_faces, 3)).astype(np.float32)
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
out_tvm = tvm.nd.empty((n_faces, 1), device=dev)
print(adj_np, idx_np.shape, idx2_np.shape)
func(adj_tvm, x_tvm, idx_tvm, out_tvm)
print(out_tvm)

exit()


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul_add(n_faces, in_feats, out_feats, dtype):
    adj = te.placeholder((n_faces, 3), name="adj", dtype=dtype)
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
task = tvm.auto_scheduler.SearchTask(func=matmul_add,
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
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
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
