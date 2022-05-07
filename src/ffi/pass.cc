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
#include <pass/gpu/simplex_buffers.h>
#include <pass/grad.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/make_1d_var.h>
#include <pass/make_const_shape.h>
#include <pass/make_parallel_reduction.h>
#include <pass/make_reduction.h>
#include <pass/merge_and_hoist_if.h>
#include <pass/move_out_first_or_last_iter.h>
#include <pass/output_intermediates.h>
#include <pass/prop_one_time_use.h>
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
    py::enum_<GradTapeMode>(m, "GradTapeMode")
        .value("All", GradTapeMode::All)
        .value("Nothing", GradTapeMode::Nothing)
        .value("NoReuseOnly", GradTapeMode::NoReuseOnly);

    m.def(
        "grad",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<ID, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &,
                const std::unordered_set<ID> &)>(&grad),
        "stmt"_a, "requires"_a, "provides"_a, "tapes"_a);
    m.def(
        "grad",
        static_cast<
            std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<ID, std::string>> (*)(
                const Stmt &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &,
                const std::unordered_set<ID> &)>(&grad),
        "func"_a, "requires"_a, "provides"_a, "tapes"_a);
    m.def(
        "grad",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<ID, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, GradTapeMode)>(&grad),
        "stmt"_a, "requires"_a, "provides"_a,
        "tape_mode"_a = GradTapeMode::NoReuseOnly);
    m.def(
        "grad",
        static_cast<
            std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<ID, std::string>> (*)(
                const Stmt &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, GradTapeMode)>(&grad),
        "func"_a, "requires"_a, "provides"_a,
        "tape_mode"_a = GradTapeMode::NoReuseOnly);

    // std::unordered_map<Load, Expr> cannot be exported to Python
    m.def(
        "output_intermediates",
        [](const Stmt &op, const std::unordered_set<ID> &intermediates) {
            return std::get<0>(outputIntermediates(op, intermediates));
        },
        "stmt"_a, "intermediates"_a);

    m.def("simplify", static_cast<Func (*)(const Func &)>(&simplify), "func"_a);
    m.def("simplify", static_cast<Stmt (*)(const Stmt &)>(&simplify), "stmt"_a);

    m.def("z3_simplify", static_cast<Func (*)(const Func &)>(&z3Simplify),
          "func"_a);
    m.def("z3_simplify", static_cast<Stmt (*)(const Stmt &)>(&z3Simplify),
          "stmt"_a);

    m.def("float_simplify", static_cast<Func (*)(const Func &)>(&floatSimplify),
          "func"_a);
    m.def("float_simplify", static_cast<Stmt (*)(const Stmt &)>(&floatSimplify),
          "stmt"_a);

    m.def("flatten_stmt_seq",
          static_cast<Func (*)(const Func &)>(&flattenStmtSeq), "func"_a);
    m.def("flatten_stmt_seq",
          static_cast<Stmt (*)(const Stmt &)>(&flattenStmtSeq), "stmt"_a);

    m.def("move_out_first_or_last_iter",
          static_cast<Func (*)(const Func &)>(&moveOutFirstOrLastIter),
          "func"_a);
    m.def("move_out_first_or_last_iter",
          static_cast<Stmt (*)(const Stmt &)>(&moveOutFirstOrLastIter),
          "stmt"_a);

    m.def("scalar_prop_const",
          static_cast<Func (*)(const Func &)>(&scalarPropConst), "func"_a);
    m.def("scalar_prop_const",
          static_cast<Stmt (*)(const Stmt &)>(&scalarPropConst), "stmt"_a);

    m.def("sink_var", static_cast<Func (*)(const Func &)>(&sinkVar), "func"_a);
    m.def("sink_var", static_cast<Stmt (*)(const Stmt &)>(&sinkVar), "stmt"_a);

    m.def("shrink_var", static_cast<Func (*)(const Func &)>(&shrinkVar),
          "func"_a);
    m.def("shrink_var", static_cast<Stmt (*)(const Stmt &)>(&shrinkVar),
          "stmt"_a);

    m.def("shrink_for", static_cast<Func (*)(const Func &)>(&shrinkFor),
          "func"_a);
    m.def("shrink_for", static_cast<Stmt (*)(const Stmt &)>(&shrinkFor),
          "stmt"_a);

    m.def("merge_and_hoist_if",
          static_cast<Func (*)(const Func &)>(&mergeAndHoistIf), "func"_a);
    m.def("merge_and_hoist_if",
          static_cast<Stmt (*)(const Stmt &)>(&mergeAndHoistIf), "stmt"_a);

    m.def("make_reduction", static_cast<Func (*)(const Func &)>(&makeReduction),
          "func"_a);
    m.def("make_reduction", static_cast<Stmt (*)(const Stmt &)>(&makeReduction),
          "stmt"_a);

    m.def("make_parallel_reduction",
          static_cast<Func (*)(const Func &)>(&makeParallelReduction),
          "func"_a);
    m.def("make_parallel_reduction",
          static_cast<Stmt (*)(const Stmt &)>(&makeParallelReduction),
          "stmt"_a);

    m.def("tensor_prop_const",
          static_cast<Func (*)(const Func &)>(&tensorPropConst), "func"_a);
    m.def("tensor_prop_const",
          static_cast<Stmt (*)(const Stmt &)>(&tensorPropConst), "stmt"_a);

    m.def("prop_one_time_use",
          static_cast<Func (*)(const Func &)>(&propOneTimeUse), "func"_a);
    m.def("prop_one_time_use",
          static_cast<Stmt (*)(const Stmt &)>(&propOneTimeUse), "stmt"_a);

    m.def("remove_writes",
          static_cast<Func (*)(const Func &, const ID &)>(&removeWrites),
          "func"_a, "single_def_id"_a = "");
    m.def("remove_writes",
          static_cast<Stmt (*)(const Stmt &, const ID &)>(&removeWrites),
          "stmt"_a, "single_def_id"_a = "");

    m.def("remove_dead_var",
          static_cast<Func (*)(const Func &)>(&removeDeadVar), "func"_a);
    m.def("remove_dead_var",
          static_cast<Stmt (*)(const Stmt &)>(&removeDeadVar), "stmt"_a);

    m.def("make_const_shape",
          static_cast<Func (*)(const Func &, const std::vector<MemType> &)>(
              &makeConstShape),
          "func"_a, "mtypes"_a);
    m.def("make_const_shape",
          static_cast<Stmt (*)(const Stmt &, const std::vector<MemType> &)>(
              &makeConstShape),
          "stmt"_a, "mtypes"_a);

    m.def("make_1d_var", static_cast<Func (*)(const Func &)>(&make1dVar),
          "func"_a);
    m.def("make_1d_var", static_cast<Stmt (*)(const Stmt &)>(&make1dVar),
          "stmt"_a);

    m.def("use_builtin_div",
          static_cast<Func (*)(const Func &)>(&useBuiltinDiv), "func"_a);
    m.def("use_builtin_div",
          static_cast<Stmt (*)(const Stmt &)>(&useBuiltinDiv), "stmt"_a);

    m.def("hoist_var_over_stmt_seq",
          static_cast<Func (*)(const Func &)>(&hoistVarOverStmtSeq), "func"_a);
    m.def("hoist_var_over_stmt_seq",
          static_cast<Stmt (*)(const Stmt &)>(&hoistVarOverStmtSeq), "stmt"_a);

    // CPU
    m.def("cpu_lower_parallel_reduction",
          static_cast<Func (*)(const Func &)>(&cpu::lowerParallelReduction));
    m.def("cpu_lower_parallel_reduction",
          static_cast<Stmt (*)(const Stmt &)>(&cpu::lowerParallelReduction));

    // GPU
    m.def("gpu_lower_parallel_reduction",
          static_cast<Func (*)(const Func &)>(&gpu::lowerParallelReduction));
    m.def("gpu_lower_parallel_reduction",
          static_cast<Stmt (*)(const Stmt &)>(&gpu::lowerParallelReduction));

    m.def("gpu_normalize_threads",
          static_cast<Func (*)(const Func &)>(&gpu::normalizeThreads),
          "func"_a);
    m.def("gpu_normalize_threads",
          static_cast<Stmt (*)(const Stmt &)>(&gpu::normalizeThreads),
          "stmt"_a);

    m.def("gpu_make_sync", static_cast<Func (*)(const Func &)>(&gpu::makeSync),
          "func"_a);
    m.def("gpu_make_sync", static_cast<Stmt (*)(const Stmt &)>(&gpu::makeSync),
          "stmt"_a);

    m.def("gpu_multiplex_buffers",
          static_cast<Func (*)(const Func &)>(&gpu::multiplexBuffers),
          "func"_a);
    m.def("gpu_multiplex_buffers",
          static_cast<Stmt (*)(const Stmt &)>(&gpu::multiplexBuffers),
          "stmt"_a);

    m.def("gpu_simplex_buffers",
          static_cast<Func (*)(const Func &)>(&gpu::simplexBuffers), "func"_a);
    m.def("gpu_simplex_buffers",
          static_cast<Stmt (*)(const Stmt &)>(&gpu::simplexBuffers), "stmt"_a);

    m.def("gpu_lower_vector",
          static_cast<Func (*)(const Func &)>(&gpu::lowerVector), "func"_a);
    m.def("gpu_lower_vector",
          static_cast<Stmt (*)(const Stmt &)>(&gpu::lowerVector), "stmt"_a);

    m.def("lower",
          static_cast<Func (*)(const Func &, const Ref<Target> &,
                               const std::unordered_set<std::string> &, int)>(
              &lower),
          "func"_a, "target"_a = nullptr,
          "skip_passes"_a = std::unordered_set<std::string>{}, "verbose"_a = 0);
    m.def("lower",
          static_cast<Stmt (*)(const Stmt &, const Ref<Target> &,
                               const std::unordered_set<std::string> &, int)>(
              &lower),
          "stmt"_a, "target"_a = nullptr,
          "skip_passes"_a = std::unordered_set<std::string>{}, "verbose"_a = 0);
}

} // namespace freetensor
