#include <ffi.h>
#include <schedule.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_schedule(py::module_ &m) {
    py::enum_<MoveToSide>(m, "MoveToSide")
        .value("Before", MoveToSide::Before)
        .value("After", MoveToSide::After);

    py::class_<Schedule>(m, "Schedule")
        .def(py::init<const Stmt &>())
        .def("ast", &Schedule::ast)
        .def("find", &Schedule::find)
        .def("split", &Schedule::split, "id"_a, "factor"_a = -1,
             "nparts"_a = -1)
        .def("reorder", &Schedule::reorder, "order"_a)
        .def("merge", &Schedule::merge, "loop1"_a, "loop2"_a)
        .def("fission", &Schedule::fission, "loop"_a, "after"_a,
             "suffix0"_a = ".a", "suffix1"_a = ".b")
        .def("fuse", &Schedule::fuse, "loop0"_a, "loop1"_a)
        .def("swap", &Schedule::swap, "order"_a)
        .def("cache", &Schedule::cache, "stmt"_a, "var"_a, "mtype"_a)
        .def("cache_reduction", &Schedule::cacheReduction, "stmt"_a, "var"_a,
             "mtype"_a)
        .def("move_to", &Schedule::moveTo, "stmt"_a, "side"_a, "dst"_a)
        .def("parallelize", &Schedule::parallelize, "loop"_a, "parallel"_a)
        .def("unroll", &Schedule::unroll, "loop"_a);
}

} // namespace ir

