#ifndef FFI_H
#define FFI_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ref.h>

PYBIND11_DECLARE_HOLDER_TYPE(T, ir::Ref<T>);

namespace ir {

namespace py = pybind11;

void init_ffi_except(py::module_ &m);
void init_ffi_tensor(py::module_ &m);
void init_ffi_buffer(py::module_ &m);
void init_ffi_ast(py::module_ &m);
void init_ffi_cursor(py::module_ &m);
void init_ffi_schedule(py::module_ &m);
void init_ffi_pass(py::module_ &m);
void init_ffi_codegen(py::module_ &m);
void init_ffi_driver(py::module_ &m);

} // namespace ir

#endif // FFI_H
