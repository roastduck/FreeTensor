#include <vector>

#include <driver/device.h>
#include <driver/device_t.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_device(py::module_ &m) {
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
        .def("compute_capability",
             [](const Ref<GPU> &target) -> std::optional<std::pair<int, int>> {
                 return target->computeCapability();
             })
        .def(
            "set_compute_capability",
            [](const Ref<GPU> &target, int major, int minor) {
                target->setComputeCapability(major, minor);
            },
            "major"_a, "minor"_a);

    py::class_<Device, Ref<Device>>(m, "Device")
        .def(py::init<const Ref<Target> &, size_t>(), "target"_a, "num"_a = 0)
        .def("target", &Device::target)
        .def("main_mem_type", &Device::mainMemType)
        .def("sync", &Device::sync)
        .def("__eq__", [](const Ref<Device> &lhs, const Ref<Device> &rhs) {
            return *lhs == *rhs;
        });

    py::class_<Target_t, Ref<Target_t>> pyTarget_t(m, "Target_t");
    pyTarget_t
        .def("type", [](const Ref<Target_t> &target) { return target->type(); })
        .def(
            "set_use_native_arch",
            [](const Ref<Target_t> &target, bool useNativeArch) {
                target->setUseNativeArch(useNativeArch);
            },
            "useNativeArch"_a = true)
        .def("use_native_arch", &Target_t::useNativeArch)
        .def("__str__", &Target_t::toString)
        .def("main_mem_type", &Target_t::mainMemType);

#ifdef FT_WITH_CUDA
    pyTarget_t.def("set_info_arch", &Target_t::setInfoArch)
        .def("info_arch", &Target_t::infoArch);
#endif // FT_WITH_CUDA

    py::class_<CPU_t, Ref<CPU_t>>(m, "CPU_t", pyTarget_t)
        .def(py::init([](bool useNativeArch) {
                 return Ref<CPU_t>::make(useNativeArch);
             }),
             "use_native_arch"_a = true);
    py::class_<GPU_t, Ref<GPU_t>>(m, "GPU_t", pyTarget_t)
        .def(py::init([](bool useNativeArch) {
                 return Ref<GPU_t>::make(useNativeArch);
             }),
             "use_native_arch"_a = true);

    py::class_<Device_t, Ref<Device_t>>(m, "Device_t")
        .def(py::init<const TargetType &, size_t>(), "targetType"_a,
             "num"_a = 0)
        .def(py::init<const TargetType &, const std::string &>(),
             "targetType"_a, "getDeviceByName"_a)
        .def("type", &Device_t::type)
        .def("num", &Device_t::num)
        .def("target", &Device_t::target);
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
