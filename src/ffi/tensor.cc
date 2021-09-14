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

    py::class_<TensorData, Ref<TensorData>>(m, "TensorData")
        .def(py::init([](py::array_t<int32_t, py::array::c_style> &np) {
            auto ndim = np.ndim();
            auto size = np.size();
            auto shapePtr = np.shape();
            auto dataPtr = np.unchecked().data();
            return Ref<TensorData>::make(
                TensorData(std::vector<int>(shapePtr, shapePtr + ndim),
                           std::vector<int>(dataPtr, dataPtr + size)));
        }))
        .def(py::init([](py::array_t<int64_t, py::array::c_style> &np) {
            auto ndim = np.ndim();
            auto size = np.size();
            auto shapePtr = np.shape();
            auto dataPtr = np.unchecked().data();
            return Ref<TensorData>::make(
                TensorData(std::vector<int>(shapePtr, shapePtr + ndim),
                           std::vector<int>(dataPtr, dataPtr + size)));
        }))
        .def(py::init([](py::array_t<float, py::array::c_style> &np) {
            auto ndim = np.ndim();
            auto size = np.size();
            auto shapePtr = np.shape();
            auto dataPtr = np.unchecked().data();
            return Ref<TensorData>::make(
                TensorData(std::vector<int>(shapePtr, shapePtr + ndim),
                           std::vector<double>(dataPtr, dataPtr + size)));
        }))
        .def(py::init([](py::array_t<double, py::array::c_style> &np) {
            auto ndim = np.ndim();
            auto size = np.size();
            auto shapePtr = np.shape();
            auto dataPtr = np.unchecked().data();
            return Ref<TensorData>::make(
                TensorData(std::vector<int>(shapePtr, shapePtr + ndim),
                           std::vector<double>(dataPtr, dataPtr + size)));
        }));
}

} // namespace ir

