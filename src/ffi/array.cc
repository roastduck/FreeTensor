#include <pybind11/numpy.h>
#include <vector>

#include <driver/array.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_array(py::module_ &m) {
#define INIT_FROM_NUMPY_DEV(nativeType, dtype)                                 \
    py::init([](py::array_t<nativeType, py::array::c_style> &np,               \
                const Ref<Device> &device) {                                   \
        std::vector<size_t> shape(np.shape(), np.shape() + np.ndim());         \
        Array arr(shape, dtype, device);                                       \
        arr.fromCPU(np.unchecked().data(), np.nbytes());                       \
        return arr;                                                            \
    })

#define INIT_FROM_NUMPY(nativeType, dtype)                                     \
    py::init([](py::array_t<nativeType, py::array::c_style> &np) {             \
        std::vector<size_t> shape(np.shape(), np.shape() + np.ndim());         \
        Array arr(shape, dtype);                                               \
        arr.fromCPU(np.unchecked().data(), np.nbytes());                       \
        return arr;                                                            \
    })

    py::class_<Array, Ref<Array>>(m, "Array")
        .def(INIT_FROM_NUMPY_DEV(double, DataType::Float64))
        .def(INIT_FROM_NUMPY_DEV(float, DataType::Float32))
        .def(INIT_FROM_NUMPY_DEV(int64_t, DataType::Int64))
        .def(INIT_FROM_NUMPY_DEV(int32_t, DataType::Int32))
        .def(INIT_FROM_NUMPY_DEV(bool, DataType::Bool))
        .def(INIT_FROM_NUMPY(double, DataType::Float64))
        .def(INIT_FROM_NUMPY(float, DataType::Float32))
        .def(INIT_FROM_NUMPY(int64_t, DataType::Int64))
        .def(INIT_FROM_NUMPY(int32_t, DataType::Int32))
        .def(INIT_FROM_NUMPY(bool, DataType::Bool))
        .def("numpy",
             [](Array &arr) -> py::object {
                 switch (arr.dtype()) {
                 case DataType::Int64: {
                     py::array_t<int64_t, py::array::c_style> np(arr.shape());
                     arr.toCPU(np.mutable_unchecked().mutable_data(),
                               np.nbytes());
                     return std::move(np); // construct an py::object by move
                 }
                 case DataType::Int32: {
                     py::array_t<int32_t, py::array::c_style> np(arr.shape());
                     arr.toCPU(np.mutable_unchecked().mutable_data(),
                               np.nbytes());
                     return std::move(np); // construct an py::object by move
                 }
                 case DataType::Float64: {
                     py::array_t<double, py::array::c_style> np(arr.shape());
                     arr.toCPU(np.mutable_unchecked().mutable_data(),
                               np.nbytes());
                     return std::move(np);
                 }
                 case DataType::Float32: {
                     py::array_t<float, py::array::c_style> np(arr.shape());
                     arr.toCPU(np.mutable_unchecked().mutable_data(),
                               np.nbytes());
                     return std::move(np);
                 }
                 case DataType::Bool: {
                     py::array_t<bool, py::array::c_style> np(arr.shape());
                     arr.toCPU(np.mutable_unchecked().mutable_data(),
                               np.nbytes());
                     return std::move(np);
                 }
                 default:
                     ASSERT(false);
                 }
             })
        .def_property_readonly("shape", &Array::shape)
        .def_property_readonly("dtype", &Array::dtype)
        .def_property_readonly("device", &Array::device);

    py::implicitly_convertible<py::array, Array>();
}
} // namespace freetensor