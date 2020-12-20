#include <buffer.h>
#include <ffi.h>

namespace ir {

void init_ffi_buffer(py::module_ &m) {
    py::enum_<AccessType>(m, "AccessType")
        .value("Input", AccessType::Input)
        .value("Output", AccessType::Output)
        .value("InOut", AccessType::InOut)
        .value("Cache", AccessType::Cache);

    py::class_<Buffer> buffer(m, "Buffer");
    buffer.def(py::init<const Tensor &, AccessType>());
}

} // namespace ir

