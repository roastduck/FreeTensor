#include <cstdio>
#include <unistd.h>

#include <config.h>
#include <ffi.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_config(py::module_ &m) {
    Config::setPrettyPrint(isatty(fileno(stdout)));

    m.def("with_mkl", Config::withMKL);
    m.def("set_pretty_print", Config::setPrettyPrint, "flag"_a = true);
    m.def("pretty_print", Config::prettyPrint);
    m.def("set_print_all_id", Config::setPrintAllId, "flag"_a = true);
    m.def("print_all_id", Config::printAllId);
}

} // namespace ir
