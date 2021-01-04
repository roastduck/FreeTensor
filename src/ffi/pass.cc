#include <ffi.h>
#include <pass/code_gen_c.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>

namespace ir {

void init_ffi_pass(py::module_ &m) {
    m.def("codeGenC", &codeGenC);
    m.def("simplifyPass", &simplifyPass);
    m.def("flattenStmtSeq", &flattenStmtSeq);
    m.def("sinkVar", &sinkVar);
    m.def("shrinkVar", &shrinkVar);
}

} // namespace ir

