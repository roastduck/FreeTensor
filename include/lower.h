#ifndef IR_LOWER_H
#define IR_LOWER_H

#include <driver/target.h>
#include <pass/gpu/correct_shared.h>
#include <pass/gpu/lower_vector.h>
#include <pass/gpu/make_sync.h>
#include <pass/gpu/normalize_threads.h>
#include <pass/make_1d_var.h>
#include <pass/make_atomic.h>
#include <pass/make_const_shape.h>
#include <pass/merge_and_hoist_if.h>
#include <pass/move_out_first_or_last_iter.h>
#include <pass/prop_const.h>
#include <pass/remove_dead_var.h>
#include <pass/remove_writes.h>
#include <pass/shrink_for.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <pass/use_builtin_div.h>

namespace ir {

template <class T> T lower(const T &t, const Ref<Target> &target) {
    T func = t;
    func = simplifyPass(func);
    func = moveOutFirstOrLastIter(func);
    func = sinkVar(func);
    func = shrinkVar(func);
    func = mergeAndHoistIf(func);
    func = propConst(func);
    func = removeWrites(func);  // After seperate_tail
    func = removeDeadVar(func); // After remove_writes and prop_const
    func = shrinkFor(func);     // After seperate_tail and remove_writes

    if (target.isValid()) {
        switch (target->type()) {
        case TargetType::GPU:
            // TODO: Support dynamic shared memory size, but the size should be
            // determined outside of kernels
            func = gpu::correctShared(func);
            func =
                makeConstShape(func, {MemType::GPUShared, MemType::GPULocal});

            // After gpu_make_sync and gpu_correct_shared. Otherwise, these 2
            // passes cannot get the right thread info
            func = gpu::normalizeThreads(func);

            func = makeAtomic(func);    // After gpu_correct_shared
            func = gpu::makeSync(func); // After gpu_normalize_threads
            func = make1dVar(func);
            func = gpu::lowerVector(func); // After make_1d_var
            break;

        case TargetType::CPU:
            func = makeAtomic(func);
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
