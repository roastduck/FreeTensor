#include <ffi.h>
#include <type/access_type.h>
#include <type/data_type.h>
#include <type/mem_type.h>

namespace freetensor {

void init_ffi_tensor_types(py::module_ &m) {
    py::class_<AccessType>(m, "AccessType")
        .def(py::init<AccessType>())
        .def(py::init(&parseAType))
        .def("__str__",
             static_cast<std::string (*)(const AccessType &)>(&toString))
        .def("__hash__", [](AccessType atype) { return (size_t)atype; })
        .def("__eq__",
             [](AccessType lhs, AccessType rhs) { return lhs == rhs; })
        .def("__eq__", [](AccessType lhs, const std::string &rhs) {
            return lhs == parseAType(rhs);
        });
    // no py::implicitly_convertible from str, because it fails silently

    m.def("is_writable", &isWritable);
    m.def("is_inputting", &isInputting);
    m.def("is_outputting", &isOutputting);

    m.def("add_outputting", &addOutputting);
    m.def("remove_outputting", &removeOutputting);

    py::class_<MemType>(m, "MemType")
        .def(py::init<MemType>())
        .def(py::init(&parseMType))
        .def("__str__",
             static_cast<std::string (*)(const MemType &)>(&toString))
        .def("__hash__", [](MemType mtype) { return (size_t)mtype; })
        .def("__eq__", [](MemType lhs, MemType rhs) { return lhs == rhs; })
        .def("__eq__", [](MemType lhs, const std::string &rhs) {
            return lhs == parseMType(rhs);
        });
    // no py::implicitly_convertible from str, because it fails silently

    py::class_<BaseDataType>(m, "BaseDataType")
        .def(py::init<BaseDataType>())
        .def(py::init(&parseBaseDataType))
        .def("__eq__",
             [](BaseDataType lhs, BaseDataType rhs) { return lhs == rhs; });
    py::class_<SignDataType>(m, "SignDataType")
        .def(py::init<SignDataType>())
        .def(py::init(&parseSignDataType))
        .def("__eq__",
             [](SignDataType lhs, SignDataType rhs) { return lhs == rhs; });
    py::class_<DataType>(m, "DataType")
        .def(py::init<DataType>())
        .def(py::init<BaseDataType>())
        .def(py::init(&parseDType))
        .def_property_readonly("base", &DataType::base)
        .def_property_readonly("sign", &DataType::sign)
        .def("__str__",
             static_cast<std::string (*)(const DataType &)>(&toString))
        .def("__repr__",
             [](DataType dtype) {
                 auto str = toString(dtype);
                 str[0] = toupper(str[0]);
                 return "<DataType " + str + ">";
             })
        .def("__hash__",
             [](DataType dtype) { return std::hash<DataType>{}(dtype); })
        .def("__eq__", [](DataType lhs, DataType rhs) { return lhs == rhs; })
        .def("__eq__", [](DataType lhs, const std::string &rhs) {
            return lhs == parseDType(rhs);
        });
    py::implicitly_convertible<BaseDataType, DataType>();
    // no py::implicitly_convertible from str, because it fails silently

    m.def("is_int", static_cast<bool (*)(const DataType &)>(&isInt));
    m.def("is_float", static_cast<bool (*)(const DataType &)>(&isFloat));
    m.def("is_number", static_cast<bool (*)(const DataType &)>(&isNumber));
    m.def("is_bool", static_cast<bool (*)(const DataType &)>(&isBool));
    m.def(
        "up_cast",
        static_cast<DataType (*)(const DataType &, const DataType &)>(&upCast));
}

} // namespace freetensor
