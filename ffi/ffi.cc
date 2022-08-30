#include <ffi.h>

namespace freetensor {

PYBIND11_MODULE(freetensor_ffi, m) {
    m.doc() = "Interface to FreeTensor's C++ backend";

    init_ffi_except(m);
    init_ffi_tensor_types(m);
    init_ffi_device(m);
    init_ffi_array(m);
    init_ffi_parallel_scope(m);
    init_ffi_metadata(m);

    // Introduce the AST, Func, Expr, and Stmt declarations first
    init_ffi_ast(m);
    init_ffi_ast_func(m);

    // FrontendVar depends on Expr declaration
    init_ffi_frontend(m);

    // Expr depends on FrontendVar
    init_ffi_ast_expr(m);

    // ReduceItem depends on Expr declaration
    init_ffi_for_property(m);
    // Tensor depends on Expr declaration
    init_ffi_tensor(m);
    // Buffer depends on Tensor declaration
    init_ffi_buffer(m);
    // Stmt depends on Tensor declaration
    init_ffi_ast_stmt(m);

    init_ffi_analyze(m);
    init_ffi_codegen(m);
    init_ffi_driver(m);
    init_ffi_debug(m);

    // Lower and Schedule depends on Target in driver
    init_ffi_autograd(m);
    init_ffi_pass(m);
    init_ffi_schedule(m);
    init_ffi_auto_schedule(m);
    init_ffi_config(m);
}

} // namespace freetensor
