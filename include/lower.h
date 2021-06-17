//
// Created by hitonami on 2021/6/15.
//

#ifndef IR_LOWER_H
#define IR_LOWER_H
#include <driver/target.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <pass/shrink_var.h>
#include <pass/merge_and_hoist_if.h>
#include <pass/seperate_tail.h>
#include <pass/remove_writes.h>
#include <pass/remove_dead_var.h>
#include <pass/shrink_for.h>
#include <pass/make_atomic.h>
#include <pass/make_const_shape.h>
#include <pass/gpu/correct_shared.h>
#include <pass/gpu/normalize_threads.h>
#include <pass/gpu/make_sync.h>
#include <pass/make_1d_var.h>
#include <pass/gpu/lower_vector.h>
#include <pass/use_builtin_div.h>

namespace ir {

template<class T>
T lower(const T &t, const Ref<Target> &target) {
    T func = t;
    func = simplifyPass(func);
    func = sinkVar(func);
    func = shrinkVar(func);
    func = mergeAndHoistIf(func);
    func = seperateTail(func);
    func = removeWrites(func);
    func = removeDeadVar(func);
    func = shrinkFor(func);

    if (target.isValid()) {
        if (target->type() == TargetType::GPU) {
            func = makeConstShape(func,{MemType::GPUShared, MemType::GPULocal} );
            func = gpu::correctShared(func);
            func = gpu::normalizeThreads(func);
            func = makeAtomic(func);
            func = gpu::makeSync(func);
            func = make1dVar(func);
            func = gpu::lowerVector(func);
        }
        else {
            func = makeAtomic(func);
        }
    }

    func = useBuiltinDiv(func);
    return func;
}

} // namespace ir

#endif // IR_LOWER_H
