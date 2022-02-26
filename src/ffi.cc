#include <ffi.h>

namespace ir {

PYBIND11_MODULE(ffi, m) {
    init_ffi_except(m);
    init_ffi_data_type(m);
    init_ffi_tensor(m);
    init_ffi_buffer(m);
    init_ffi_frontend(m);
    init_ffi_ast(m);
    init_ffi_cursor(m);
    init_ffi_schedule(m);
    init_ffi_analyze(m);
    init_ffi_pass(m);
    init_ffi_codegen(m);
    init_ffi_driver(m);
    init_ffi_debug(m);
    init_ffi_auto_schedule(m);
    init_ffi_config(m);
}

} // namespace ir
