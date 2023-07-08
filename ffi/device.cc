#include <vector>

#include <driver/device.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_device(py::module_ &m) {
    py::enum_<TargetType>(m, "TargetType")
        .value("CPU", TargetType::CPU)
        .value("GPU", TargetType::GPU);

    py::class_<Target, Ref<Target>> pyTarget(m, "Target");
    py::class_<Device, Ref<Device>> pyDevice(m, "Device");

    pyTarget
        .def(py::init(
            [](const Ref<Device> &device) { return device->target(); }))
        .def("type", [](const Ref<Target> &target) { return target->type(); })
        .def("__str__", &Target::toString)
        .def("main_mem_type", &Target::mainMemType)
        .def("use_native_arch", &Target::useNativeArch)
        .def("__eq__",
             static_cast<bool (*)(const Ref<Target> &, const Ref<Target> &)>(
                 &isSameTarget));

    py::class_<CPUTarget, Ref<CPUTarget>>(m, "CPUTarget", pyTarget)
        .def("set_use_native_arch", &CPUTarget::setUseNativeArch,
             "use_native_arch"_a = true)
        .def("n_cores", &CPUTarget::nCores);

#ifdef FT_WITH_CUDA
    py::class_<GPUTarget, Ref<GPUTarget>>(m, "GPUTarget", pyTarget)
        .def("compute_capability", &GPUTarget::computeCapability)
        .def("warp_size", &GPUTarget::warpSize)
        .def("multi_processor_count", &GPUTarget::multiProcessorCount);
#endif // FT_WITH_CUDA

    pyDevice
        .def(py::init<const TargetType &, int>(), "target_type"_a, "num"_a = 0)
        .def(py::init<const TargetType &, const std::string &>(),
             "target_type"_a, "get_device_by_name"_a)
        .def(py::init<const TargetType &, const std::string &, size_t>(),
             "target_type"_a, "get_device_by_full_name"_a, "nth"_a)
        .def("type", &Device::type)
        .def("num", &Device::num)
        .def("target", &Device::target)
        .def("main_mem_type", &Device::mainMemType)
        .def("sync", &Device::sync)
        .def("__eq__", [](const Ref<Device> &lhs, const Ref<Device> &rhs) {
            return *lhs == *rhs;
        });
    py::implicitly_convertible<Device, Target>();
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
            type = &typeid(freetensor::CPUTarget);
            return static_cast<const freetensor::CPUTarget *>(src);
#ifdef FT_WITH_CUDA
        case freetensor::TargetType::GPU:
            type = &typeid(freetensor::GPUTarget);
            return static_cast<const freetensor::GPUTarget *>(src);
#endif // FT_WITH_CUDA
        default:
            ERROR("Unexpected target type");
        }
    }
};

} // namespace pybind11
