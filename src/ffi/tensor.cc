#include <ffi.h>
#include <tensor.h>

namespace ir {

void init_ffi_tensor(py::module_ &m) {
    py::class_<Tensor> tensor(m, "Tensor");
    tensor.def(py::init<const std::vector<Expr> &, DataType>())
        .def(py::init([](const Expr &e, DataType d) { return Tensor({e}, d); }))
        .def_property_readonly(
            "shape", static_cast<const std::vector<Expr> &(Tensor::*)() const>(
                         &Tensor::shape))
        .def_property_readonly("dtype", &Tensor::dtype);
}

} // namespace ir

