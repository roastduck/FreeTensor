#include <ffi.h>
#include <pass/code_gen_c.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>

namespace ir {

void init_ffi_pass(py::module_ &m) {
    m.def("code_gen_c", &codeGenC);
    m.def("simplify_pass", &simplifyPass);
    m.def("flatten_stmt_seq", &flattenStmtSeq);
    m.def("sink_var", &sinkVar);
    m.def("shrink_var", &shrinkVar);
}

} // namespace ir

