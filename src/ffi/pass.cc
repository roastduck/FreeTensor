#include <ffi.h>
#include <pass/code_gen_c.h>
#include <pass/print_pass.h>

namespace ir {

void init_ffi_pass(py::module_ &m) {
    m.def("printPass", &printPass);
    m.def("codeGenC", &codeGenC);
}

} // namespace ir

