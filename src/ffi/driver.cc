#include <pybind11/numpy.h>
#include <vector>

#include <driver.h>
#include <except.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_driver(py::module_ &m) {
    py::enum_<TargetType>(m, "TargetType")
        .value("CPU", TargetType::CPU)
        .value("GPU", TargetType::GPU);

    py::class_<Target, Ref<Target>> pyTarget(m, "Target");
    pyTarget
        .def("type", [](const Ref<Target> &target) { return target->type(); })
        .def(
            "set_use_native_arch",
            [](const Ref<Target> &target, bool useNativeArch) {
                target->setUseNativeArch(useNativeArch);
            },
            "useNativeArch"_a = true)
        .def("use_native_arch", &Target::useNativeArch)
        .def("__str__", &Target::toString)
        .def("main_mem_type", &Target::mainMemType);
    py::class_<CPU, Ref<CPU>>(m, "CPU", pyTarget)
        .def(py::init([](bool useNativeArch) {
                 return Ref<CPU>::make(useNativeArch);
             }),
             "use_native_arch"_a = true);
    py::class_<GPU, Ref<GPU>>(m, "GPU", pyTarget)
        .def(py::init([](bool useNativeArch) {
                 return Ref<GPU>::make(useNativeArch);
             }),
             "use_native_arch"_a = true)
        .def(
            "set_compute_capability",
            [](const Ref<GPU> &target, int major, int minor) {
                target->setComputeCapability(major, minor);
            },
            "major"_a, "minor"_a);
    // We can't export .compute_capability() to Python due to an issue in
    // PyBind11

    py::class_<Device>(m, "Device")
        .def(py::init<const Ref<Target> &, size_t>(), "target"_a, "num"_a = 0)
        .def("target", &Device::target)
        .def("main_mem_type", &Device::mainMemType)
        .def("sync", &Device::sync);

    py::class_<Array, Ref<Array>>(m, "Array")
        .def(py::init([](py::array_t<double, py::array::c_style> &np,
                         const Device &device) {
            std::vector<size_t> shape(np.shape(), np.shape() + np.ndim());
            Array arr(shape, DataType::Float64, device);
            arr.fromCPU(np.unchecked().data(), np.nbytes());
            return arr;
        }))
        .def(py::init([](py::array_t<float, py::array::c_style> &np,
                         const Device &device) {
            std::vector<size_t> shape(np.shape(), np.shape() + np.ndim());
            Array arr(shape, DataType::Float32, device);
            arr.fromCPU(np.unchecked().data(), np.nbytes());
            return arr;
        }))
        .def(py::init([](py::array_t<int64_t, py::array::c_style> &np,
                         const Device &device) {
            std::vector<size_t> shape(np.shape(), np.shape() + np.ndim());
            Array arr(shape, DataType::Int64, device);
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
        .def(py::init([](py::array_t<bool, py::array::c_style> &np,
                         const Device &device) {
            std::vector<size_t> shape(np.shape(), np.shape() + np.ndim());
            Array arr(shape, DataType::Bool, device);
            arr.fromCPU(np.unchecked().data(), np.nbytes());
            return arr;
        }))
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

    py::class_<Driver, Ref<Driver>>(m, "Driver")
        .def(py::init<const Func &, const std::string &, const Device &>())
        .def("set_params",
             static_cast<void (Driver::*)(
                 const std::vector<Ref<Array>> &,
                 const std::unordered_map<std::string, Ref<Array>> &)>(
                 &Driver::setParams),
             "args"_a, "kws"_a = std::unordered_map<std::string, Ref<Array>>())
        .def("set_params",
             static_cast<void (Driver::*)(
                 const std::unordered_map<std::string, Ref<Array>> &)>(
                 &Driver::setParams),
             "kws"_a)
        .def("run", &Driver::run)
        .def("sync", &Driver::sync)
        .def("collect_returns", &Driver::collectReturns)
        .def("time", &Driver::time, "rounds"_a = 10, "warmpups"_a = 3);
}

} // namespace freetensor

namespace pybind11 {

template <> struct polymorphic_type_hook<freetensor::Target> {
    static const void *get(const freetensor::Target *src,
                           const std::type_info *&type) {
        if (src == nullptr) {
            return src;
        }
        switch (src->type()) {
        case freetensor::TargetType::CPU:
            type = &typeid(freetensor::CPU);
            return static_cast<const freetensor::CPU *>(src);
        case freetensor::TargetType::GPU:
            type = &typeid(freetensor::GPU);
            return static_cast<const freetensor::GPU *>(src);
        default:
            ERROR("Unexpected target type");
        }
    }
};

} // namespace pybind11
