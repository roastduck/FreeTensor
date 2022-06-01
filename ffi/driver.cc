#include <pybind11/numpy.h>
#include <vector>

#include <driver.h>
#include <except.h>
#include <ffi.h>

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
                 &Driver::setArgs),
             "kws"_a)
        .def("run", &Driver::run)
        .def("sync", &Driver::sync)
        .def("collect_returns", &Driver::collectReturns)
        .def("time", &Driver::time, "rounds"_a = 10, "warmpups"_a = 3);
}

} // namespace freetensor
