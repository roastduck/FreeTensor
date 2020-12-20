#include <ffi.h>

namespace ir {

PYBIND11_MODULE(ffi, m) { init_ffi_ast(m); }

} // namespace ir

