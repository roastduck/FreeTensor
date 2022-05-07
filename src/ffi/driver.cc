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
        .def("main_mem_type", &Target::mainMemType)
        .def("__eq__",
             static_cast<bool (*)(const Ref<Target> &, const Ref<Target> &)>(
                 &isSame));
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

    py::class_<Device, Ref<Device>>(m, "Device")
        .def(py::init<const Ref<Target> &, size_t>(), "target"_a, "num"_a = 0)
        .def("target", &Device::target)
        .def("main_mem_type", &Device::mainMemType)
        .def("sync", &Device::sync);

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

    py::class_<Driver, Ref<Driver>>(m, "Driver")
        .def(py::init<const Func &, const std::string &, const Ref<Device> &>())
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
