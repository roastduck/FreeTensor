#include <except.h>
#include <ffi.h>

namespace freetensor {

void init_ffi_except(py::module_ &m) {
    py::register_exception<Error>(m, "Error");
    py::register_exception<InvalidSchedule>(m, "InvalidSchedule");
    py::register_exception<InvalidProgram>(m, "InvalidProgram");
    py::register_exception<DriverError>(m, "DriverError");
    py::register_exception<AssertAlwaysFalse>(m, "AssertAlwaysFalse");

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const InterruptExcept &e) {
            PyErr_SetInterrupt();
            PyErr_CheckSignals();
            throw py::error_already_set();
        }
    });
}

} // namespace freetensor
