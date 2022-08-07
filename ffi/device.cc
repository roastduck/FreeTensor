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
    pyTarget
        .def("type", [](const Ref<Target> &target) { return target->type(); })
        .def(
            "set_use_native_arch",
            [](const Ref<Target> &target, bool useNativeArch) {
                target->setUseNativeArch(useNativeArch);
            },
            "use_native_arch"_a = true)
        .def("use_native_arch", &Target::useNativeArch)
        .def("__str__", &Target::toString)
        .def("main_mem_type", &Target::mainMemType)
        .def("__eq__",
             static_cast<bool (*)(const Ref<Target> &, const Ref<Target> &)>(
                 &isSameTarget));

    py::class_<CPU, Ref<CPU>>(m, "CPU", pyTarget)
        .def(py::init([](bool useNativeArch) {
                 return Ref<CPU>::make(useNativeArch);
             }),
             "use_native_arch"_a = true);

#ifdef FT_WITH_CUDA
    py::class_<GPU, Ref<GPU>>(m, "GPU", pyTarget)
        .def(py::init([](bool useNativeArch) {
                 return Ref<GPU>::make(nullptr, useNativeArch);
             }),
             "use_native_arch"_a = true)
        .def("info_arch", &GPU::infoArch);
#endif // FT_WITH_CUDA

    py::class_<Device, Ref<Device>>(m, "Device")
        .def(py::init<const TargetType &, size_t>(), "target_type"_a,
             "num"_a = 0)
        .def(py::init<const TargetType &, const std::string &>(),
             "target_type"_a, "get_device_by_name"_a)
        .def(py::init<const TargetType &, const std::string &, int>(),
             "target_type"_a, "get_device_by_full_name"_a, "nth"_a)
        .def("type", &Device::type)
        .def("num", &Device::num)
        .def("target", &Device::target)
        .def("sync", &Device::sync)
        .def("__eq__", [](const Ref<Device> &lhs, const Ref<Device> &rhs) {
            return *lhs == *rhs;
        });
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
