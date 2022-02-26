#ifndef IR_LOWER_H
#define IR_LOWER_H

#include <driver/target.h>
#include <pass/cpu/lower_parallel_reduction.h>
#include <pass/float_simplify.h>
#include <pass/gpu/lower_parallel_reduction.h>
#include <pass/gpu/lower_vector.h>
#include <pass/gpu/make_sync.h>
#include <pass/gpu/multiplex_buffers.h>
#include <pass/gpu/normalize_threads.h>
#include <pass/gpu/simplex_buffers.h>
#include <pass/make_1d_var.h>
#include <pass/make_const_shape.h>
#include <pass/make_parallel_reduction.h>
#include <pass/merge_and_hoist_if.h>
#include <pass/move_out_first_or_last_iter.h>
#include <pass/prop_one_time_use.h>
#include <pass/remove_cyclic_assign.h>
#include <pass/remove_dead_var.h>
#include <pass/remove_writes.h>
#include <pass/scalar_prop_const.h>
#include <pass/shrink_for.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <pass/tensor_prop_const.h>
#include <pass/use_builtin_div.h>

namespace ir {

template <class T> T lower(const T &t, const Ref<Target> &target) {
    T func = t;
    func = scalarPropConst(func);
    func = removeDeadVar(func);
    func = propOneTimeUse(func);
    func = floatSimplify(func); // After propOneTimeUse
    func = simplifyPass(func);
    func = moveOutFirstOrLastIter(func);
    func = sinkVar(func);
    func = shrinkVar(func);
    func = mergeAndHoistIf(func);
    func = tensorPropConst(func);
    func = removeWrites(func);
    func = removeCyclicAssign(func); // After remove_writes
    func = removeDeadVar(func);      // After remove_writes and prop_const
    func = makeParallelReduction(func);
    func = shrinkFor(func); // After remove_writes and make_parallel_reduction

    if (target.isValid()) {
        switch (target->type()) {
        case TargetType::GPU:
            // Before gpu_nromalize_threads
            func = gpu::lowerParallelReduction(func);

            // TODO: Support dynamic shared memory size, but the size should be
            // determined outside of kernels
            func = gpu::multiplexBuffers(func);
            func = gpu::simplexBuffers(func);
            // FIXME: MemType::GPUGlobal should also be make const, but only
            // inside a kernel
            func =
                makeConstShape(func, std::vector<MemType>{MemType::GPUShared,
                                                          MemType::GPULocal});
            func = gpu::normalizeThreads(func); // After gpu_multiplex_buffers
            func = gpu::makeSync(func);         // After gpu_normalize_threads
            func = make1dVar(func);
            func = gpu::lowerVector(func); // After make_1d_var
            break;

        case TargetType::CPU:
            func = cpu::lowerParallelReduction(func);
            break;

        default:
            ASSERT(false);
        }
    }

    // After passes including architecture-specific ones
    func = useBuiltinDiv(func);

    return func;
}

} // namespace ir

#endif // IR_LOWER_H
