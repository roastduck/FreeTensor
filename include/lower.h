#ifndef IR_LOWER_H
#define IR_LOWER_H

#include <driver/target.h>
#include <pass/float_simplify.h>
#include <pass/gpu/correct_shared_and_global.h>
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
    func = floatSimplify(func);
    std::cout << "Finish #1" << std::endl;
    func = simplifyPass(func);
    std::cout << "Finish #2" << std::endl;
    func = moveOutFirstOrLastIter(func);
    std::cout << "Finish #3" << std::endl;
    func = sinkVar(func);
    std::cout << "Finish #4" << std::endl;
    func = shrinkVar(func);
    std::cout << "Finish #5" << std::endl;
    func = mergeAndHoistIf(func);
    std::cout << "Finish #6" << std::endl;
    func = propConst(func);
    std::cout << "Finish #7" << std::endl;
    func = removeWrites(func); // After seperate_tail
    std::cout << "Finish #8" << std::endl;
    func = removeDeadVar(func); // After remove_writes and prop_const
    std::cout << "Finish #9" << std::endl;
    func = shrinkFor(func); // After seperate_tail and remove_writes
    std::cout << "Finish #10" << std::endl;

    if (target.isValid()) {
        switch (target->type()) {
        case TargetType::GPU:
            // TODO: Support dynamic shared memory size, but the size should be
            // determined outside of kernels
            func = gpu::correctSharedAndGlobal(func);
            // FIXME: MemType::GPUGlobal should also be make const, but only
            // inside a kernel
            func =
                makeConstShape(func, std::vector<MemType>{MemType::GPUShared,
                                                          MemType::GPULocal});

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
