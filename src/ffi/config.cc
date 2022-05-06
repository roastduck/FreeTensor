#include <config.h>
#include <driver/device.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_config(py::module_ &m) {
    Config::init();

    m.def("with_mkl", Config::withMKL);
    m.def("with_cuda", Config::withCUDA);
    m.def("set_pretty_print", Config::setPrettyPrint, "flag"_a = true);
    m.def("pretty_print", Config::prettyPrint);
    m.def("set_print_all_id", Config::setPrintAllId, "flag"_a = true);
    m.def("print_all_id", Config::printAllId);
    m.def("set_werror", Config::setWerror, "flag"_a = true);
    m.def("werror", Config::werror);
    m.def("set_default_device", Config::setDefaultDevice, "device"_a);
    m.def("default_device", Config::defaultDevice);
}

} // namespace freetensor
