#include <pybind11/numpy.h>

#include <ffi.h>
#include <ref.h>
#include <tensor.h>

namespace ir {

void init_ffi_tensor(py::module_ &m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<Expr> &, DataType>())
        .def(py::init([](const Expr &e, DataType d) { return Tensor({e}, d); }))
        .def_property_readonly("shape",
                               [](const Tensor &t) {
                                   return std::vector<Expr>(t.shape().begin(),
                                                            t.shape().end());
                               })
        .def_property_readonly("dtype", &Tensor::dtype);
}

} // namespace ir
