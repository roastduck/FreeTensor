#include <debug.h>
#include <ffi.h>

namespace freetensor {

void init_ffi_debug(py::module_ &m) {
    py::class_<Logger>(m, "Logger")
        .def("enable", &Logger::enable)
        .def("disable", &Logger::disable);
    m.def("logger", &logger, py::return_value_policy::reference);
}

} // namespace freetensor
