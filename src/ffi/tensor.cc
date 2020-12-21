#include <ffi.h>
#include <tensor.h>

namespace ir {

void init_ffi_tensor(py::module_ &m) {
    py::enum_<DataType>(m, "DataType")
        .value("Float32", DataType::Float32)
        .value("Int32", DataType::Int32);

    py::class_<Tensor> tensor(m, "Tensor");
    tensor.def(py::init<const std::vector<Expr> &, DataType>());
}

} // namespace ir

