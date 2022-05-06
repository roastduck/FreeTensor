#include <buffer.h>
#include <ffi.h>

namespace freetensor {

void init_ffi_buffer(py::module_ &m) {
    py::class_<AccessType>(m, "AccessType")
        .def(py::init<AccessType>())
        .def(py::init(&parseAType))
        .def("__str__", static_cast<std::string (*)(AccessType)>(&toString))
        .def("__hash__", [](AccessType atype) { return (size_t)atype; })
        .def("__eq__",
             [](AccessType lhs, AccessType rhs) { return lhs == rhs; });
    // no py::implicitly_convertible, because it fails silently

    py::class_<MemType>(m, "MemType")
        .def(py::init<MemType>())
        .def(py::init(&parseMType))
        .def("__str__", static_cast<std::string (*)(MemType)>(&toString))
        .def("__hash__", [](MemType mtype) { return (size_t)mtype; })
        .def("__eq__", [](MemType lhs, MemType rhs) { return lhs == rhs; });
    // no py::implicitly_convertible, because it fails silently

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
