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

    py::class_<Target, Ref<Target>> pyTarget(m, "Target");
    pyTarget
        .def("type", [](const Ref<Target> &target) { return target->type(); })
        .def(
            "set_use_native_arch",
            [](const Ref<Target> &target, bool useNativeArch) {
                target->setUseNativeArch(useNativeArch);
            },
            "useNativeArch"_a = true)
        .def("use_native_arch",
             [](const Ref<Target> &target) { return target->useNativeArch(); })
        .def("__str__",
             [](const Ref<Target> &target) { return target->toString(); });
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
        .def("target", &Device::target);

    py::class_<Array>(m, "Array")
        .def(py::init([](py::array_t<float, py::array::c_style> &np,
                         const Device &device) {
            Array arr(np.size(), DataType::Float32, device);
            arr.fromCPU(np.unchecked().data(), np.nbytes());
            return arr;
        }))
        .def(py::init([](py::array_t<int32_t, py::array::c_style> &np,
                         const Device &device) {
            Array arr(np.size(), DataType::Int32, device);
            arr.fromCPU(np.unchecked().data(), np.nbytes());
            return arr;
        }))
        .def("numpy", [](Array &arr) -> py::object {
            switch (arr.dtype()) {
            case DataType::Int32: {
                std::vector<size_t> shape(1, arr.nElem());
                py::array_t<int32_t, py::array::c_style> np(shape);
                arr.toCPU(np.mutable_unchecked().mutable_data(), np.nbytes());
                return std::move(np); // construct an py::object by move
            }
            case DataType::Float32: {
                std::vector<size_t> shape(1, arr.nElem());
                py::array_t<float, py::array::c_style> np(shape);
                arr.toCPU(np.mutable_unchecked().mutable_data(), np.nbytes());
                return std::move(np);
            }
            default:
                ASSERT(false);
            }
        });

    py::class_<Driver>(m, "Driver")
        .def(py::init<const Func &, const std::string &, const Device &>())
        .def("set_params",
             static_cast<void (Driver::*)(
                 const std::vector<Array *> &,
                 const std::unordered_map<std::string, Array *> &)>(
                 &Driver::setParams),
             "args"_a, "kws"_a = std::unordered_map<std::string, Array *>())
        .def("set_params",
             static_cast<void (Driver::*)(
                 const std::unordered_map<std::string, Array *> &)>(
                 &Driver::setParams),
             "kws"_a)
        .def("run", &Driver::run)
        .def("time", &Driver::time, "rounds"_a = 10, "warmpups"_a = 3);
}

} // namespace ir

namespace pybind11 {

template <> struct polymorphic_type_hook<ir::Target> {
    static const void *get(const ir::Target *src, const std::type_info *&type) {
        if (src == nullptr) {
            return src;
        }
        switch (src->type()) {
        case ir::TargetType::CPU:
            type = &typeid(ir::CPU);
            return static_cast<const ir::CPU *>(src);
        case ir::TargetType::GPU:
            type = &typeid(ir::GPU);
            return static_cast<const ir::GPU *>(src);
        default:
            ERROR("Unexpected target type");
        }
    }
};

} // namespace pybind11

