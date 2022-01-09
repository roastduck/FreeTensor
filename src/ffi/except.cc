#include <except.h>
#include <ffi.h>

namespace ir {

void init_ffi_except(py::module_ &m) {
    py::register_exception<Error>(m, "Error");
    py::register_exception<InvalidSchedule>(m, "InvalidSchedule");
    py::register_exception<InvalidProgram>(m, "InvalidProgram");
    py::register_exception<DriverError>(m, "DriverError");
    py::register_exception<AssertAlwaysFalse>(m, "AssertAlwaysFalse");
}

} // namespace ir

