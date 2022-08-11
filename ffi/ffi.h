#ifndef FREE_TENSOR_FFI_H
#define FREE_TENSOR_FFI_H

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ref.h>

PYBIND11_DECLARE_HOLDER_TYPE(T, freetensor::Ref<T>);

namespace freetensor {

namespace py = pybind11;

void init_ffi_except(py::module_ &m);
void init_ffi_tensor_types(py::module_ &m);
void init_ffi_device(py::module_ &m);
void init_ffi_array(py::module_ &m);
void init_ffi_tensor(py::module_ &m);
void init_ffi_parallel_scope(py::module_ &m);
void init_ffi_for_property(py::module_ &m);
void init_ffi_buffer(py::module_ &m);
void init_ffi_frontend(py::module_ &m);
void init_ffi_metadata(py::module_ &m);

void init_ffi_ast(py::module_ &m);
void init_ffi_ast_func(py::module_ &m);
void init_ffi_ast_expr(py::module_ &m);
void init_ffi_ast_stmt(py::module_ &m);

void init_ffi_schedule(py::module_ &m);
void init_ffi_analyze(py::module_ &m);
void init_ffi_autograd(py::module_ &m);
void init_ffi_pass(py::module_ &m);
void init_ffi_codegen(py::module_ &m);
void init_ffi_driver(py::module_ &m);
void init_ffi_debug(py::module_ &m);
void init_ffi_auto_schedule(py::module_ &m);
void init_ffi_config(py::module_ &m);

} // namespace freetensor

#endif // FREE_TENSOR_FFI_H
