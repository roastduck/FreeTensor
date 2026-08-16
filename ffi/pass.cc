#include <ffi.h>
#include <lower.h>
#include <pass/cpu/lower_parallel_reduction.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/float_simplify.h>
#include <pass/gpu/lower_parallel_reduction.h>
#include <pass/gpu/lower_vector.h>
#include <pass/gpu/make_sync.h>
#include <pass/gpu/multiplex_buffers.h>
#include <pass/gpu/normalize_threads.h>
#include <pass/gpu/normalize_var_in_kernel.h>
#include <pass/gpu/simplex_buffers.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/make_heap_alloc.h>
#include <pass/make_parallel_reduction.h>
#include <pass/make_reduction.h>
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

using namespace pybind11::literals;

void init_ffi_pass(py::module_ &m) {
    m.def("simplify",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &simplify<const std::optional<bool> &,
                        const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("simplify",
          static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                               const std::optional<double> &)>(&simplify),
          "stmt"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    m.def("z3_simplify",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &z3Simplify<const std::optional<bool> &,
                          const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("z3_simplify",
          static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                               const std::optional<double> &)>(&z3Simplify),
          "stmt"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    m.def("pb_simplify",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &pbSimplify<const std::optional<bool> &,
                          const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("pb_simplify",
          static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                               const std::optional<double> &)>(&pbSimplify),
          "stmt"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    m.def("float_simplify",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &floatSimplify<const std::optional<bool> &,
                             const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("float_simplify",
          static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                               const std::optional<double> &)>(&floatSimplify),
          "stmt"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    m.def("flatten_stmt_seq",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &flattenStmtSeq<const std::optional<bool> &,
                              const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("flatten_stmt_seq",
          static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                               const std::optional<double> &)>(&flattenStmtSeq),
          "stmt"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    m.def("move_out_first_or_last_iter",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &moveOutFirstOrLastIter<const std::optional<bool> &,
                                      const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("move_out_first_or_last_iter",
          static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &moveOutFirstOrLastIter),
          "stmt"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    m.def("scalar_prop_const",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &scalarPropConst<const std::optional<bool> &,
                               const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def(
        "scalar_prop_const",
        static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                             const std::optional<double> &)>(&scalarPropConst),
        "stmt"_a, "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt);

    m.def("sink_var",
          static_cast<Func (*)(
              const Func &, const std::optional<std::unordered_set<ID>> &,
              const std::function<bool(const Stmt &)> &,
              const std::optional<bool> &, const std::optional<double> &)>(
              &sinkVar<const std::optional<std::unordered_set<ID>> &,
                       const std::function<bool(const Stmt &)> &,
                       const std::optional<bool> &,
                       const std::optional<double> &>),
          "func"_a, "to_sink"_a = std::nullopt, "scope_filter"_a = nullptr,
          "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt);
    m.def("sink_var",
          static_cast<Stmt (*)(
              const Stmt &, const std::optional<std::unordered_set<ID>> &,
              const std::function<bool(const Stmt &)> &,
              const std::optional<bool> &, const std::optional<double> &)>(
              &sinkVar),
          "stmt"_a, "to_sink"_a = std::nullopt, "scope_filter"_a = nullptr,
          "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt);

    m.def("shrink_var",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &shrinkVar<const std::optional<bool> &,
                         const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("shrink_var",
          static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                               const std::optional<double> &)>(&shrinkVar),
          "stmt"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    m.def("shrink_for",
          static_cast<Func (*)(const Func &, const ID &, const bool &,
                               const bool &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &shrinkFor<const ID &, const bool &, const bool &,
                         const std::optional<bool> &,
                         const std::optional<double> &>),
          "func"_a, py::arg_v("sub_ast", ID(), "ID()"), "do_simplify"_a = true,
          "unordered"_a = false, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("shrink_for",
          static_cast<Stmt (*)(const Stmt &, const ID &, bool, bool,
                               const std::optional<bool> &,
                               const std::optional<double> &)>(&shrinkFor),
          "stmt"_a, py::arg_v("sub_ast", ID(), "ID()"), "do_simplify"_a = true,
          "unordered"_a = false, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    m.def("merge_and_hoist_if",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &mergeAndHoistIf<const std::optional<bool> &,
                               const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def(
        "merge_and_hoist_if",
        static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                             const std::optional<double> &)>(&mergeAndHoistIf),
        "stmt"_a, "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt);

    m.def("make_reduction",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &makeReduction<const std::optional<bool> &,
                             const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("make_reduction",
          static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                               const std::optional<double> &)>(&makeReduction),
          "stmt"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    m.def("make_parallel_reduction",
          static_cast<Func (*)(const Func &, const Ref<Target> &,
                               const std::optional<bool> &,
                               const std::optional<double> &)>(
              &makeParallelReduction<const Ref<Target> &,
                                     const std::optional<bool> &,
                                     const std::optional<double> &>),
          "func"_a, "target"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("make_parallel_reduction",
          static_cast<Stmt (*)(
              const Stmt &, const Ref<Target> &, const std::optional<bool> &,
              const std::optional<double> &)>(&makeParallelReduction),
          "stmt"_a, "target"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    m.def("tensor_prop_const",
          static_cast<Func (*)(const Func &, const ID &, const ID &,
                               const std::optional<bool> &,
                               const std::optional<double> &)>(
              &tensorPropConst<const ID &, const ID &,
                               const std::optional<bool> &,
                               const std::optional<double> &>),
          "func"_a, py::arg_v("both_in_sub_ast", ID(), "ID()"),
          py::arg_v("either_in_sub_ast", ID(), "ID()"),
          "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt);
    m.def("tensor_prop_const",
          static_cast<Stmt (*)(
              const Stmt &, const ID &, const ID &, const std::optional<bool> &,
              const std::optional<double> &)>(&tensorPropConst),
          "stmt"_a, py::arg_v("both_in_sub_ast", ID(), "ID()"),
          py::arg_v("either_in_sub_ast", ID(), "ID()"),
          "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt);

    m.def("prop_one_time_use",
          static_cast<Func (*)(const Func &, const ID &,
                               const std::optional<bool> &,
                               const std::optional<double> &)>(
              &propOneTimeUse<const ID &, const std::optional<bool> &,
                              const std::optional<double> &>),
          "func"_a, py::arg_v("sub_ast", ID(), "ID()"),
          "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt);
    m.def("prop_one_time_use",
          static_cast<Stmt (*)(const Stmt &, const ID &,
                               const std::optional<bool> &,
                               const std::optional<double> &)>(&propOneTimeUse),
          "stmt"_a, py::arg_v("sub_ast", ID(), "ID()"),
          "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt);

    m.def("remove_writes",
          static_cast<Func (*)(const Func &, const ID &,
                               const std::optional<bool> &,
                               const std::optional<double> &)>(
              &removeWrites<const ID &, const std::optional<bool> &,
                            const std::optional<double> &>),
          "func"_a, py::arg_v("single_def_id", ID(), "ID()"),
          "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt);
    m.def("remove_writes",
          static_cast<Stmt (*)(const Stmt &, const ID &,
                               const std::optional<bool> &,
                               const std::optional<double> &)>(&removeWrites),
          "stmt"_a, py::arg_v("single_def_id", ID(), "ID()"),
          "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt);

    m.def("remove_cyclic_assign",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &removeCyclicAssign<const std::optional<bool> &,
                                  const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("remove_cyclic_assign",
          static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &removeCyclicAssign),
          "stmt"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    m.def("remove_dead_var",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &removeDeadVar<const std::optional<bool> &,
                             const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("remove_dead_var",
          static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                               const std::optional<double> &)>(&removeDeadVar),
          "stmt"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    m.def("make_heap_alloc",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &makeHeapAlloc<const std::optional<bool> &,
                             const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("make_heap_alloc",
          static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                               const std::optional<double> &)>(&makeHeapAlloc),
          "stmt"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    m.def("use_builtin_div",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &useBuiltinDiv<const std::optional<bool> &,
                             const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("use_builtin_div",
          static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                               const std::optional<double> &)>(&useBuiltinDiv),
          "stmt"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    m.def("hoist_var_over_stmt_seq",
          static_cast<Func (*)(
              const Func &, const std::optional<std::vector<ID>> &,
              const std::optional<bool> &, const std::optional<double> &)>(
              &hoistVarOverStmtSeq),
          "func"_a, "together_ids"_a = std::nullopt,
          "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt);
    m.def("hoist_var_over_stmt_seq",
          static_cast<Stmt (*)(
              const Stmt &, const std::optional<std::vector<ID>> &,
              const std::optional<bool> &, const std::optional<double> &)>(
              &hoistVarOverStmtSeq),
          "stmt"_a, "together_ids"_a = std::nullopt,
          "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt);

    // CPU
    m.def("cpu_lower_parallel_reduction",
          static_cast<Func (*)(const Func &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &cpu::lowerParallelReduction<const std::optional<bool> &,
                                           const std::optional<double> &>),
          "func"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);
    m.def("cpu_lower_parallel_reduction",
          static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                               const std::optional<double> &)>(
              &cpu::lowerParallelReduction),
          "stmt"_a, "as_subprocess"_a = std::nullopt,
          "timeout"_a = std::nullopt);

    // GPU
#ifdef FT_WITH_CUDA
#define GPU_ONLY(name, ...) name, __VA_ARGS__
#else
#define GPU_ONLY(name, ...)                                                    \
    name, [](const py::args &, const py::kwargs &) {                           \
        ERROR(FT_MSG << name                                                   \
                     << " is unavailable because FT_WITH_CUDA is disabled "    \
                        "when building FreeTensor");                           \
    }
#endif

    m.def(GPU_ONLY(
        "gpu_lower_parallel_reduction",
        static_cast<Func (*)(const Func &, const std::optional<bool> &,
                             const std::optional<double> &)>(
            &gpu::lowerParallelReduction<const std::optional<bool> &,
                                         const std::optional<double> &>),
        "func"_a, "as_subprocess"_a = std::nullopt,
        "timeout"_a = std::nullopt));
    m.def(
        GPU_ONLY("gpu_lower_parallel_reduction",
                 static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                                      const std::optional<double> &)>(
                     &gpu::lowerParallelReduction),
                 "stmt"_a, "as_subprocess"_a = std::nullopt,
                 "timeout"_a = std::nullopt));

    m.def(
        GPU_ONLY("gpu_normalize_threads",
                 static_cast<Func (*)(const Func &, const std::optional<bool> &,
                                      const std::optional<double> &)>(
                     &gpu::normalizeThreads<const std::optional<bool> &,
                                            const std::optional<double> &>),
                 "func"_a, "as_subprocess"_a = std::nullopt,
                 "timeout"_a = std::nullopt));
    m.def(
        GPU_ONLY("gpu_normalize_threads",
                 static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                                      const std::optional<double> &)>(
                     &gpu::normalizeThreads),
                 "stmt"_a, "as_subprocess"_a = std::nullopt,
                 "timeout"_a = std::nullopt));

    m.def(
        GPU_ONLY("gpu_normalize_var_in_kernel",
                 static_cast<Func (*)(const Func &, const std::optional<bool> &,
                                      const std::optional<double> &)>(
                     &gpu::normalizeVarInKernel<const std::optional<bool> &,
                                                const std::optional<double> &>),
                 "func"_a, "as_subprocess"_a = std::nullopt,
                 "timeout"_a = std::nullopt));
    m.def(
        GPU_ONLY("gpu_normalize_var_in_kernel",
                 static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                                      const std::optional<double> &)>(
                     &gpu::normalizeVarInKernel),
                 "stmt"_a, "as_subprocess"_a = std::nullopt,
                 "timeout"_a = std::nullopt));

    m.def(GPU_ONLY(
        "gpu_make_sync",
        static_cast<Func (*)(const Func &, const Ref<GPUTarget> &,
                             const std::optional<bool> &,
                             const std::optional<double> &)>(
            &gpu::makeSync<const Ref<GPUTarget> &, const std::optional<bool> &,
                           const std::optional<double> &>),
        "func"_a, "target"_a, "as_subprocess"_a = std::nullopt,
        "timeout"_a = std::nullopt));
    m.def(GPU_ONLY(
        "gpu_make_sync",
        static_cast<Stmt (*)(const Stmt &, const Ref<GPUTarget> &,
                             const std::optional<bool> &,
                             const std::optional<double> &)>(&gpu::makeSync),
        "stmt"_a, "target"_a, "as_subprocess"_a = std::nullopt,
        "timeout"_a = std::nullopt));

    m.def(
        GPU_ONLY("gpu_multiplex_buffers",
                 static_cast<Func (*)(const Func &, const Ref<GPUTarget> &,
                                      const ID &, const std::optional<bool> &,
                                      const std::optional<double> &)>(
                     &gpu::multiplexBuffers<const Ref<GPUTarget> &, const ID &,
                                            const std::optional<bool> &,
                                            const std::optional<double> &>),
                 "func"_a, "target"_a, py::arg_v("def_id", ID(), "ID()"),
                 "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt));
    m.def(GPU_ONLY("gpu_multiplex_buffers",
                   static_cast<Stmt (*)(const Stmt &, const Ref<GPUTarget> &,
                                        const ID &, const std::optional<bool> &,
                                        const std::optional<double> &)>(
                       &gpu::multiplexBuffers),
                   "stmt"_a, "target"_a, py::arg_v("def_id", ID(), "ID()"),
                   "as_subprocess"_a = std::nullopt,
                   "timeout"_a = std::nullopt));

    m.def(GPU_ONLY(
        "gpu_simplex_buffers",
        static_cast<Func (*)(const Func &, const ID &,
                             const std::optional<bool> &,
                             const std::optional<double> &)>(
            &gpu::simplexBuffers<const ID &, const std::optional<bool> &,
                                 const std::optional<double> &>),
        "func"_a, py::arg_v("def_id", ID(), "ID()"),
        "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt));
    m.def(GPU_ONLY("gpu_simplex_buffers",
                   static_cast<Stmt (*)(
                       const Stmt &, const ID &, const std::optional<bool> &,
                       const std::optional<double> &)>(&gpu::simplexBuffers),
                   "stmt"_a, py::arg_v("def_id", ID(), "ID()"),
                   "as_subprocess"_a = std::nullopt,
                   "timeout"_a = std::nullopt));

    m.def(
        GPU_ONLY("gpu_lower_vector",
                 static_cast<Func (*)(const Func &, const std::optional<bool> &,
                                      const std::optional<double> &)>(
                     &gpu::lowerVector<const std::optional<bool> &,
                                       const std::optional<double> &>),
                 "func"_a, "as_subprocess"_a = std::nullopt,
                 "timeout"_a = std::nullopt));
    m.def(GPU_ONLY(
        "gpu_lower_vector",
        static_cast<Stmt (*)(const Stmt &, const std::optional<bool> &,
                             const std::optional<double> &)>(&gpu::lowerVector),
        "stmt"_a, "as_subprocess"_a = std::nullopt,
        "timeout"_a = std::nullopt));

#undef GPU_ONLY

    m.def("lower",
          static_cast<Func (*)(const Func &, const Ref<Target> &,
                               const std::unordered_set<std::string> &, int,
                               const std::optional<bool> &,
                               const std::optional<double> &)>(&lower),
          "func"_a, "target"_a = nullptr,
          "skip_passes"_a = std::unordered_set<std::string>{}, "verbose"_a = 0,
          "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt);
    m.def("lower",
          static_cast<Stmt (*)(const Stmt &, const Ref<Target> &,
                               const std::unordered_set<std::string> &, int,
                               const std::optional<bool> &,
                               const std::optional<double> &)>(&lower),
          "stmt"_a, "target"_a = nullptr,
          "skip_passes"_a = std::unordered_set<std::string>{}, "verbose"_a = 0,
          "as_subprocess"_a = std::nullopt, "timeout"_a = std::nullopt);
}

} // namespace freetensor
