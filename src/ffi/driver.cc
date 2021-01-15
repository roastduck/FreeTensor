#include <pybind11/numpy.h>
#include <vector>

#include <driver.h>
#include <except.h>
#include <ffi.h>

namespace ir {

void init_ffi_driver(py::module_ &m) {
    py::class_<Array>(m, "Array")
        .def(py::init([](py::array_t<float> &np, const std::string &device) {
            std::vector<size_t> shape(np.shape(), np.shape() + np.ndim());
            Array arr(shape, DataType::Float32, device);
            arr.fromCPU(np.unchecked().data(), np.nbytes());
            return arr;
        }))
        .def(py::init([](py::array_t<int32_t> &np, const std::string &device) {
            std::vector<size_t> shape(np.shape(), np.shape() + np.ndim());
            Array arr(shape, DataType::Int32, device);
            arr.fromCPU(np.unchecked().data(), np.nbytes());
            return arr;
        }))
        .def("numpy", [](Array &arr) -> py::object {
            switch (arr.dtype()) {
            case DataType::Int32: {
                py::array_t<int32_t, py::array::c_style> np(arr.shape());
                arr.toCPU(np.mutable_unchecked().mutable_data(), np.nbytes());
                return std::move(np); // construct an py::object by move
            }
            case DataType::Float32: {
                py::array_t<float, py::array::c_style> np(arr.shape());
                arr.toCPU(np.mutable_unchecked().mutable_data(), np.nbytes());
                return std::move(np);
            }
            default:
                ASSERT(false);
            }
        });

    py::class_<Driver>(m, "Driver")
        .def(py::init<const std::string &, const std::vector<std::string> &>())
        .def("set_params", &Driver::setParams)
        .def("run", &Driver::run);
}

} // namespace ir

