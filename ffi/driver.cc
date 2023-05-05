#include <pybind11/numpy.h>
#include <vector>

#include <driver.h>
#include <except.h>
#include <ffi.h>
#include <serialize/load_driver.h>
#include <serialize/print_driver.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_driver(py::module_ &m) {
    py::class_<Driver, Ref<Driver>>(m, "Driver")
        .def(py::init<const Func &, const std::string &, const Ref<Device> &,
                      bool>())
        .def(py::init<const Func &, const std::string &, const Ref<Device> &,
                      const Ref<Device> &, bool>())
        .def("set_args",
             static_cast<void (Driver::*)(
                 const std::vector<Ref<Array>> &,
                 const std::unordered_map<std::string, Ref<Array>> &)>(
                 &Driver::setArgs),
             "args"_a, "kws"_a = std::unordered_map<std::string, Ref<Array>>())
        .def("set_args",
             static_cast<void (Driver::*)(
                 const std::unordered_map<std::string, Ref<Array>> &)>(
                 &Driver::setArgs))
        .def("run", &Driver::run)
        .def("sync", &Driver::sync)
        .def("collect_returns", &Driver::collectReturns)
        .def("time", &Driver::time, "rounds"_a = 10, "warmups"_a = 3)
        .def_property_readonly("device", &Driver::device);

    // Serialization
    m.def("load_target",
          [](const std::pair<const std::string &, const std::string &>
                 &txt_data) { return std::apply(loadTarget, txt_data); });
    m.def("load_device",
          [](const std::pair<const std::string &, const std::string &>
                 &txt_data) { return std::apply(loadDevice, txt_data); });
    m.def("load_array",
          [](const std::pair<const std::string &, const std::string &>
                 &txt_data) { return std::apply(loadArray, txt_data); });
    m.def("dump_target", [](const Ref<Target> &target_) {
        auto &&[ret_meta, ret_data] = dumpTarget(target_);
        return std::make_pair(ret_meta, py::bytes(ret_data));
    });
    m.def("dump_device", [](const Ref<Device> &device_) {
        auto &&[ret_meta, ret_data] = dumpDevice(device_);
        return std::make_pair(ret_meta, py::bytes(ret_data));
    });
    m.def("dump_array", [](const Ref<Array> &array_) {
        auto &&[ret_meta, ret_data] = dumpArray(array_);
        return std::make_pair(ret_meta, py::bytes(ret_data));
    });
}

} // namespace freetensor
