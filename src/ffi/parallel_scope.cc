#include <ffi.h>
#include <parallel_scope.h>

namespace freetensor {

void init_ffi_parallel_scope(py::module_ &m) {
    py::class_<SerialScope>(m, "SerialScope")
        .def(py::init<>())
        .def("__str__",
             [](const SerialScope &scope) { return toString(scope); })
        .def("__eq__", [](const SerialScope &lhs, const SerialScope &rhs) {
            return lhs == rhs;
        });

    py::class_<OpenMPScope>(m, "OpenMPScope")
        .def(py::init<>())
        .def("__str__",
             [](const OpenMPScope &scope) { return toString(scope); })
        .def("__eq__", [](const OpenMPScope &lhs, const OpenMPScope &rhs) {
            return lhs == rhs;
        });

    py::class_<CUDAStreamScope>(m, "CUDAStreamScope")
        .def(py::init<>())
        .def("__str__",
             [](const CUDAStreamScope &scope) { return toString(scope); })
        .def("__eq__", [](const CUDAStreamScope &lhs,
                          const CUDAStreamScope &rhs) { return lhs == rhs; });

    py::enum_<CUDAScope::Level>(m, "CUDAScopeLevel")
        .value("Block", CUDAScope::Level::Block)
        .value("Thread", CUDAScope::Level::Thread);
    py::enum_<CUDAScope::Dim>(m, "CUDAScopeDim")
        .value("X", CUDAScope::Dim::X)
        .value("Y", CUDAScope::Dim::Y)
        .value("Z", CUDAScope::Dim::Z);
    py::class_<CUDAScope>(m, "CUDAScope")
        .def(py::init(
            [](const CUDAScope::Level &level, const CUDAScope::Dim &dim) {
                return CUDAScope{level, dim};
            }))
        .def("__str__", [](const CUDAScope &scope) { return toString(scope); })
        .def("__eq__", [](const CUDAScope &lhs, const CUDAScope &rhs) {
            return lhs == rhs;
        });

    // Factory function, used as a class
    m.def("ParallelScope", &parseParallelScope);
}

} // namespace freetensor
