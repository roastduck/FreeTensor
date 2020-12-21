#include <ffi.h>
#include <pass/print_pass.h>

namespace ir {

void init_ffi_pass(py::module_ &m) { m.def("printPass", &printPass); }

} // namespace ir

