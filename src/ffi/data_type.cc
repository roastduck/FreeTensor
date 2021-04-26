#include <data_type.h>
#include <ffi.h>

namespace ir {

void init_ffi_data_type(py::module_ &m) {
    py::enum_<DataType>(m, "DataType")
        .value("Float32", DataType::Float32)
        .value("Int32", DataType::Int32)
        .value("Bool", DataType::Bool)
        .value("Void", DataType::Void);
}

} // namespace ir

