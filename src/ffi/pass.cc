#include <ffi.h>
#include <lower.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/float_simplify.h>
#include <pass/gpu/correct_shared_and_global.h>
#include <pass/gpu/lower_vector.h>
#include <pass/gpu/make_sync.h>
#include <pass/gpu/normalize_threads.h>
#include <pass/grad.h>
#include <pass/make_1d_var.h>
#include <pass/make_atomic.h>
#include <pass/make_const_shape.h>
#include <pass/make_reduction.h>
#include <pass/merge_and_hoist_if.h>
#include <pass/move_out_first_or_last_iter.h>
#include <pass/output_intermediates.h>
#include <pass/prop_const.h>
#include <pass/prop_one_time_use.h>
#include <pass/remove_dead_var.h>
#include <pass/remove_writes.h>
#include <pass/shrink_for.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <pass/use_builtin_div.h>

namespace ir {

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
                       std::unordered_map<std::string, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &)>(&grad),
        "stmt"_a, "requires"_a, "provides"_a, "tapes"_a);
    m.def(
        "grad",
        static_cast<
            std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>> (*)(
                const Stmt &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &)>(&grad),
        "func"_a, "requires"_a, "provides"_a, "tapes"_a);
    m.def(
        "grad",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, GradTapeMode)>(&grad),
        "stmt"_a, "requires"_a, "provides"_a, "tapes"_a);
    m.def(
        "grad",
        static_cast<
            std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>> (*)(
                const Stmt &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, GradTapeMode)>(&grad),
        "func"_a, "requires"_a, "provides"_a, "tapes"_a);

    // std::unordered_map<Load, Expr> cannot be exported to Python
    m.def(
        "output_intermediates",
        [](const Stmt &op,
           const std::unordered_set<std::string> &intermediates) {
            return std::get<0>(outputIntermediates(op, intermediates));
        },
        "stmt"_a, "intermediates"_a);

    m.def("simplify_pass", static_cast<Func (*)(const Func &)>(&simplifyPass),
          "func"_a);
    m.def("simplify_pass", static_cast<Stmt (*)(const Stmt &)>(&simplifyPass),
          "stmt"_a);

    m.def("float_simplify", static_cast<Func (*)(const Func &)>(&floatSimplify),
          "func"_a);
    m.def("float_simplify", static_cast<Stmt (*)(const Stmt &)>(&floatSimplify),
          "stmt"_a);

    m.def("flatten_stmt_seq",
          static_cast<Func (*)(const Func &, bool &&)>(&flattenStmtSeq),
          "func"_a, "popVarDef"_a = false);
    m.def("flatten_stmt_seq",
          static_cast<Stmt (*)(const Stmt &, bool)>(&flattenStmtSeq), "stmt"_a,
          "popVarDef"_a = false);

    m.def("move_out_first_or_last_iter",
          static_cast<Func (*)(const Func &)>(&moveOutFirstOrLastIter),
          "func"_a);
    m.def("move_out_first_or_last_iter",
          static_cast<Stmt (*)(const Stmt &)>(&moveOutFirstOrLastIter),
          "stmt"_a);

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

    m.def("make_atomic", static_cast<Func (*)(const Func &)>(&makeAtomic),
          "func"_a);
    m.def("make_atomic", static_cast<Stmt (*)(const Stmt &)>(&makeAtomic),
          "stmt"_a);

    m.def("prop_const", static_cast<Func (*)(const Func &)>(&propConst),
          "func"_a);
    m.def("prop_const", static_cast<Stmt (*)(const Stmt &)>(&propConst),
          "stmt"_a);

    m.def("prop_one_time_use",
          static_cast<Func (*)(const Func &)>(&propOneTimeUse), "func"_a);
    m.def("prop_one_time_use",
          static_cast<Stmt (*)(const Stmt &)>(&propOneTimeUse), "stmt"_a);

    m.def("remove_writes", static_cast<Func (*)(const Func &)>(&removeWrites),
          "func"_a);
    m.def("remove_writes", static_cast<Stmt (*)(const Stmt &)>(&removeWrites),
          "stmt"_a);

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

    // GPU
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

    m.def("gpu_correct_shared_and_global",
          static_cast<Func (*)(const Func &)>(&gpu::correctSharedAndGlobal),
          "func"_a);
    m.def("gpu_correct_shared_and_global",
          static_cast<Stmt (*)(const Stmt &)>(&gpu::correctSharedAndGlobal),
          "stmt"_a);

    m.def("gpu_lower_vector",
          static_cast<Func (*)(const Func &)>(&gpu::lowerVector), "func"_a);
    m.def("gpu_lower_vector",
          static_cast<Stmt (*)(const Stmt &)>(&gpu::lowerVector), "stmt"_a);

    m.def("lower",
          static_cast<Func (*)(const Func &, const Ref<Target> &)>(&lower),
          "func"_a, "target"_a = nullptr);
    m.def("lower",
          static_cast<Stmt (*)(const Stmt &, const Ref<Target> &)>(&lower),
          "stmt"_a, "target"_a = nullptr);
}

} // namespace ir

