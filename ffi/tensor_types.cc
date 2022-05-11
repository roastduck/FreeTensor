#include <buffer.h>
#include <data_type.h>
#include <ffi.h>

namespace freetensor {

void init_ffi_tensor_types(py::module_ &m) {
    py::class_<AccessType>(m, "AccessType")
        .def(py::init<AccessType>())
        .def(py::init(&parseAType))
        .def("__str__", static_cast<std::string (*)(AccessType)>(&toString))
        .def("__hash__", [](AccessType atype) { return (size_t)atype; })
        .def("__eq__",
             [](AccessType lhs, AccessType rhs) { return lhs == rhs; })
        .def("__eq__", [](AccessType lhs, const std::string &rhs) {
            return lhs == parseAType(rhs);
        });
    // no py::implicitly_convertible, because it fails silently

    py::class_<MemType>(m, "MemType")
        .def(py::init<MemType>())
        .def(py::init(&parseMType))
        .def("__str__", static_cast<std::string (*)(MemType)>(&toString))
        .def("__hash__", [](MemType mtype) { return (size_t)mtype; })
        .def("__eq__", [](MemType lhs, MemType rhs) { return lhs == rhs; })
        .def("__eq__", [](MemType lhs, const std::string &rhs) {
            return lhs == parseMType(rhs);
        });
    // no py::implicitly_convertible, because it fails silently

    py::class_<DataType>(m, "DataType")
        .def(py::init<DataType>())
        .def(py::init(&parseDType))
        .def("__str__", static_cast<std::string (*)(DataType)>(&toString))
        .def("__repr__",
             [](DataType dtype) {
                 auto str = toString(dtype);
                 str[0] = toupper(str[0]);
                 return "DataType." + str;
             })
        .def("__hash__", [](DataType dtype) { return (size_t)dtype; })
        .def("__eq__", [](DataType lhs, DataType rhs) { return lhs == rhs; })
        .def("__eq__", [](DataType lhs, const std::string &rhs) {
            return lhs == parseDType(rhs);
        });
    // no py::implicitly_convertible, because it fails silently

    m.def("is_int", &isInt);
    m.def("is_float", &isFloat);
    m.def("is_number", &isNumber);
    m.def("is_bool", &isBool);
    m.def("up_cast", &upCast);
}

} // namespace freetensor
