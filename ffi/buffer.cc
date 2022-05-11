#include <buffer.h>
#include <ffi.h>

namespace freetensor {

void init_ffi_buffer(py::module_ &m) {
    py::class_<Buffer, Ref<Buffer>> buffer(m, "Buffer");
    buffer
        .def(py::init([](const Ref<Tensor> &t, AccessType a, MemType m) {
            return makeBuffer(t, a, m);
        }))
        .def_property_readonly(
            "tensor",
            [](const Ref<Buffer> &b) -> Ref<Tensor> { return b->tensor(); })
        .def_property_readonly("atype", &Buffer::atype)
        .def_property_readonly("mtype", &Buffer::mtype);
}

} // namespace freetensor
