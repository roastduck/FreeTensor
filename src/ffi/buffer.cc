#include <buffer.h>
#include <ffi.h>

namespace ir {

void init_ffi_buffer(py::module_ &m) {
    py::enum_<AccessType>(m, "AccessType")
        .value("Input", AccessType::Input)
        .value("Output", AccessType::Output)
        .value("InOut", AccessType::InOut)
        .value("Cache", AccessType::Cache);

    py::enum_<MemType>(m, "MemType")
        .value("ByValue", MemType::ByValue)
        .value("CPU", MemType::CPU)
        .value("GPUGlobal", MemType::GPUGlobal)
        .value("GPUShared", MemType::GPUShared)
        .value("GPULocal", MemType::GPULocal);

    py::class_<Buffer, Ref<Buffer>> buffer(m, "Buffer");
    buffer.def(py::init<const Tensor &, AccessType, MemType>())
        .def_property_readonly(
            "tensor",
            static_cast<const Tensor &(Buffer::*)() const>(&Buffer::tensor))
        .def_property_readonly("atype", &Buffer ::atype)
        .def_property_readonly("mtype", &Buffer::mtype);
}

} // namespace ir

