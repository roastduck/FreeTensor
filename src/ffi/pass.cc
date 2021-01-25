#include <ffi.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/gpu/make_sync.h>
#include <pass/shrink_for.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_pass(py::module_ &m) {
    m.def("simplify_pass", &simplifyPass, "ast"_a);
    m.def("flatten_stmt_seq", &flattenStmtSeq, "ast"_a, "popVarDef"_a = false);
    m.def("sink_var", &sinkVar, "ast"_a);
    m.def("shrink_var", &shrinkVar, "ast"_a);
    m.def("shrink_for", &shrinkFor, "ast"_a);

    // GPU
    m.def("gpu_make_sync", &gpu::makeSync, "ast"_a);
}

} // namespace ir

