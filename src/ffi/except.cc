#include <except.h>
#include <ffi.h>

namespace ir {

void init_ffi_except(py::module_ &m) {
    py::register_exception<Error>(m, "Error");
    py::register_exception<InvalidSchedule>(m, "InvalidSchedule");
}

} // namespace ir

