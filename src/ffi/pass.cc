#include <ffi.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/gpu/correct_shared.h>
#include <pass/gpu/make_sync.h>
#include <pass/gpu/normalize_threads.h>
#include <pass/make_1d_var.h>
#include <pass/make_atomic.h>
#include <pass/make_const_shape.h>
#include <pass/make_reduction.h>
#include <pass/merge_and_hoist_if.h>
#include <pass/remove_writes.h>
#include <pass/seperate_tail.h>
#include <pass/shrink_for.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <pass/use_builtin_div.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_pass(py::module_ &m) {
    m.def("simplify_pass", &simplifyPass, "ast"_a);
    m.def("flatten_stmt_seq", &flattenStmtSeq, "ast"_a, "popVarDef"_a = false);
    m.def("sink_var", &sinkVar, "ast"_a);
    m.def("shrink_var", &shrinkVar, "ast"_a);
    m.def("shrink_for", &shrinkFor, "ast"_a, "keepConst"_a = false);
    m.def("merge_and_hoist_if", &mergeAndHoistIf, "ast"_a);
    m.def("seperate_tail", &seperateTail, "ast"_a);
    m.def("make_reduction", &makeReduction, "ast"_a);
    m.def("make_atomic", &makeAtomic, "ast"_a);
    m.def("remove_writes", &removeWrites, "ast"_a);
    m.def("make_const_shape", &makeConstShape, "ast"_a, "mtypes"_a);
    m.def("make_1d_var", &make1dVar, "ast"_a);
    m.def("use_builtin_div", &useBuiltinDiv, "ast"_a);

    // GPU
    m.def("gpu_normalize_threads", &gpu::normalizeThreads, "ast"_a);
    m.def("gpu_make_sync", &gpu::makeSync, "ast"_a);
    m.def("gpu_correct_shared", &gpu::correctShared, "ast"_a);
}

} // namespace ir

