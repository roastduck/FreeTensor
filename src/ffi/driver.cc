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
        .def("__str__",
             [](const Ref<Target> &target) { return target->toString(); });
    py::class_<CPU, Ref<CPU>>(m, "CPU", pyTarget).def(py::init([]() {
        return Ref<CPU>::make();
    }));
    py::class_<GPU, Ref<GPU>>(m, "GPU", pyTarget)
        .def(py::init([]() { return Ref<GPU>::make(); }))
        .def("set_compute_capability",
             [](const Ref<GPU> &target, int value) {
                 target->setComputeCapability(value);
             })
        .def("compute_capability", [](const Ref<GPU> &target) {
            return target->computeCapability();
        });

    py::class_<Device>(m, "Device")
        .def(py::init<const Ref<Target> &, size_t>(), "target"_a, "num"_a = 0);

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
        .def(py::init<const std::string &, const std::vector<std::string> &,
                      const Device &>())
        .def("set_params", &Driver::setParams)
        .def("run", &Driver::run);
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

