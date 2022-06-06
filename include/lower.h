#ifndef FREE_TENSOR_LOWER_H
#define FREE_TENSOR_LOWER_H

#include <unordered_set>

#include <config.h>
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
#include <pass/z3_simplify.h>

namespace freetensor {

/**
 * Lower an AST using a series of passes
 *
 * @param ast : The AST to be lowered. Can be a `Func` or a `Stmt`
 * @param target : Lower the AST to a target with target-specific
 * passes, then the AST can be used for codegen. If not set, use the default
 * Target in Config
 * @param skipPasses : Skip some pass for testing or debugging. Names in
 * `skipPasses` are in underscore_style, as in Python. Please note that some
 * passes will not be skipped even specified in these parameter, because they
 * are indirectly called in some other passes
 * @param verbose : 0 = print nothing. 1 = print the lowered AST. 2 = print AST
 * after every single passes
 */
template <class T>
T lower(const T &_ast, const Ref<Target> &_target = nullptr,
        const std::unordered_set<std::string> &skipPasses = {},
        int verbose = 0) {

    auto target = _target.isValid() ? _target : Config::defaultTarget();

    auto maybePrint = [&](const std::string &name, const T &ast) -> T {
        if (verbose >= 2) {
            logger() << "AST after " << name << " is:" << std::endl
                     << toString(ast) << std::endl;
        }
        return ast;
    };

#define FIRST_OF(x, ...) (x)
#define APPLY(name, pass, ...)                                                 \
    skipPasses.count(name) ? FIRST_OF(__VA_ARGS__)                             \
                           : maybePrint(name, pass(__VA_ARGS__))

    T ast = _ast;
    ast = APPLY("scalar_prop_const", scalarPropConst, ast);
    ast = APPLY("remove_dead_var", removeDeadVar, ast);
    ast = APPLY("prop_one_time_use", propOneTimeUse, ast);
    ast = APPLY("float_simplify", floatSimplify, ast); // After propOneTimeUse
    ast = APPLY("z3_simplify", z3Simplify, ast);
    ast = APPLY("simplify", simplify, ast);
    ast = APPLY("move_out_first_or_last_iter", moveOutFirstOrLastIter, ast);
    ast = APPLY("sink_var", sinkVar, ast);
    ast = APPLY("shrink_var", shrinkVar, ast);
    ast = APPLY("merge_and_hoist_if", mergeAndHoistIf, ast);
    ast = APPLY("tensor_prop_const", tensorPropConst, ast);
    ast = APPLY("remove_writes", removeWrites, ast);
    ast = APPLY("remove_cyclic_assign", removeCyclicAssign,
                ast); // After remove_writes
    ast = APPLY("remove_dead_var", removeDeadVar,
                ast); // After remove_writes and prop_const
    ast = APPLY("make_parallel_reduction", makeParallelReduction, ast);
    ast = APPLY("shrink_for", shrinkFor,
                ast); // After remove_writes and make_parallel_reduction

    switch (target->type()) {
    case TargetType::GPU:
        // Before gpu_nromalize_threads
        ast = APPLY("gpu_lower_parallel_reduction", gpu::lowerParallelReduction,
                    ast);

        // TODO: Support dynamic shared memory size, but the size should be
        // determined outside of kernels
        ast = APPLY("gpu_multiplex_buffers", gpu::multiplexBuffers, ast);
        ast = APPLY("gpu_simplex_buffers", gpu::simplexBuffers, ast);
        // FIXME: MemType::GPUGlobal should also be make const, but only
        // inside a kernel
        ast =
            APPLY("make_const_shape", makeConstShape, ast,
                  std::vector<MemType>{MemType::GPUShared, MemType::GPULocal});
        ast = APPLY("gpu_normalize_threads", gpu::normalizeThreads,
                    ast); // After gpu_multiplex_buffers
        ast = APPLY("gpu_make_sync", gpu::makeSync,
                    ast); // After gpu_normalize_threads
        ast = APPLY("make_1d_var", make1dVar,
                    ast); // FIXME: make1dVar will break the shape of
                          // returned tensors
        ast = APPLY("gpu_lower_vector", gpu::lowerVector,
                    ast); // After make_1d_var
        ast = APPLY("use_builtin_div", useBuiltinDiv, ast);
        break;

    case TargetType::CPU:
        ast = APPLY("cpu_lower_parallel_reduction", cpu::lowerParallelReduction,
                    ast);
        ast = APPLY("use_builtin_div", useBuiltinDiv, ast);
        break;

    default:
        ASSERT(false);
    }

#undef FIRST_OF
#undef APPLY

    if (verbose >= 1) {
        logger() << "The lowered AST is:" << std::endl
                 << toString(ast) << std::endl;
    }

    return ast;
}

} // namespace freetensor

#endif // FREE_TENSOR_LOWER_H
