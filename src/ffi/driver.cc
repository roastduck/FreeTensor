#include <pybind11/numpy.h>
#include <vector>

#include <driver.h>
#include <except.h>
#include <ffi.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_driver(py::module_ &m) {
    py::enum_<TargetType>(m, "TargetType")
        .value("CPU", TargetType::CPU)
        .value("GPU", TargetType::GPU);

    py::class_<Ref<Target>>(m, "Target")
        .def(py::init<const Ref<CPU> &>())
        .def(py::init<const Ref<GPU> &>())
        .def("type", [](const Ref<Target> &target) { return target->type(); })
        .def("__str__",
             [](const Ref<Target> &target) {
                 return target.isValid() ? target->toString() : "";
             })
        .def("__repr__", [](const Ref<Target> &target) {
            return target.isValid() ? "<Target: " + target->toString() + ">"
                                    : "None";
        });
    py::class_<Ref<CPU>>(m, "CPU")
        .def(py::init([]() { return Ref<CPU>::make(); }))
        .def("type", [](const Ref<CPU> &target) { return target->type(); })
        .def("__str__",
             [](const Ref<CPU> &target) {
                 return target.isValid() ? target->toString() : "";
             })
        .def("__repr__", [](const Ref<CPU> &target) {
            return target.isValid() ? "<CPU: " + target->toString() + ">"
                                    : "None";
        });
    py::class_<Ref<GPU>>(m, "GPU")
        .def(py::init([]() { return Ref<GPU>::make(); }))
        .def("type", [](const Ref<GPU> &target) { return target->type(); })
        .def("__str__",
             [](const Ref<GPU> &target) {
                 return target.isValid() ? target->toString() : "";
             })
        .def("__repr__", [](const Ref<GPU> &target) {
            return target.isValid() ? "<GPU: " + target->toString() + ">"
                                    : "None";
        });
    py::implicitly_convertible<Ref<CPU>, Ref<Target>>();
    py::implicitly_convertible<Ref<GPU>, Ref<Target>>();

    py::class_<Device>(m, "Device")
        .def(py::init<const Ref<Target> &, size_t>(), "target"_a, "num"_a = 0);

    py::class_<Array>(m, "Array")
        .def(py::init([](py::array_t<float, py::array::c_style> &np,
                         const Device &device) {
            std::vector<size_t> shape(np.shape(), np.shape() + np.ndim());
            Array arr(shape, DataType::Float32, device);
            arr.fromCPU(np.unchecked().data(), np.nbytes());
            return arr;
        }))
        .def(py::init([](py::array_t<int32_t, py::array::c_style> &np,
                         const Device &device) {
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

