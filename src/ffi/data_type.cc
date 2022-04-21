#include <data_type.h>
#include <ffi.h>

namespace ir {

void init_ffi_data_type(py::module_ &m) {
    py::class_<DataType>(m, "DataType")
        .def(py::init<DataType>())
        .def(py::init(&parseDType))
        .def("__str__", static_cast<std::string (*)(DataType)>(&toString))
        .def("__hash__", [](DataType dtype) { return (size_t)dtype; })
        .def("__eq__", [](DataType lhs, DataType rhs) { return lhs == rhs; });
    // no py::implicitly_convertible, because it fails silently

    m.def("is_int", &isInt);
    m.def("is_float", &isFloat);
    m.def("is_number", &isNumber);
    m.def("is_bool", &isBool);
    m.def("up_cast", &upCast);
}

} // namespace ir
