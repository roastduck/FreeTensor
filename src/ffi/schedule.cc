#include <auto_schedule/auto_schedule.h>
#include <driver/array.h>
#include <ffi.h>
#include <schedule.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_schedule(py::module_ &m) {
    py::enum_<MoveToSide>(m, "MoveToSide")
        .value("Before", MoveToSide::Before)
        .value("After", MoveToSide::After);
    py::enum_<VarSplitMode>(m, "VarSplitMode")
        .value("FixedSize", VarSplitMode::FixedSize)
        .value("RelaxedSize", VarSplitMode::RelaxedSize);

    py::class_<Schedule>(m, "Schedule")
        .def(py::init<const Stmt &>())
        .def(py::init<const Func &>())
        .def("ast", &Schedule::ast)
        .def("func", &Schedule::func)
        .def("logs", &Schedule::logs)
        .def("find", static_cast<Cursor (Schedule::*)(
                         const std::function<bool(const Cursor &)> &) const>(
                         &Schedule::find))
        .def("find",
             static_cast<Cursor (Schedule::*)(const std::string &) const>(
                 &Schedule::find))
        .def("find_all",
             static_cast<std::vector<Cursor> (Schedule::*)(
                 const std::function<bool(const Cursor &)> &) const>(
                 &Schedule::findAll))
        .def("find_all", static_cast<std::vector<Cursor> (Schedule::*)(
                             const std::string &) const>(&Schedule::findAll))
        .def("split", &Schedule::split, "id"_a, "factor"_a = -1,
             "nparts"_a = -1)
        .def("reorder", &Schedule::reorder, "order"_a)
        .def("merge", &Schedule::merge, "loop1"_a, "loop2"_a)
        .def("fission", &Schedule::fission, "loop"_a, "after"_a,
             "suffix0"_a = ".a", "suffix1"_a = ".b")
        .def("fuse", &Schedule::fuse, "loop0"_a, "loop1"_a)
        .def("swap", &Schedule::swap, "order"_a)
        .def("blend", &Schedule::blend, "loop"_a)
        .def("cache", &Schedule::cache, "stmt"_a, "var"_a, "mtype"_a)
        .def("cache_reduction", &Schedule::cacheReduction, "stmt"_a, "var"_a,
             "mtype"_a)
        .def("set_mem_type", &Schedule::setMemType, "def"_a, "mtype"_a)
        .def("var_split", &Schedule::varSplit, "vardef"_a, "dim"_a, "mode"_a,
             "factor"_a = -1, "nparts"_a = -1)
        .def("var_reorder", &Schedule::varReorder, "vardef"_a, "order"_a)
        .def("move_to", &Schedule::moveTo, "stmt"_a, "side"_a, "dst"_a)
        .def("inline", &Schedule::inlining, "vardef"_a)
        .def("parallelize", &Schedule::parallelize, "loop"_a, "parallel"_a)
        .def("unroll", &Schedule::unroll, "loop"_a, "immedate"_a = false)
        .def("vectorize", &Schedule::vectorize, "loop"_a)
        .def("seperate_tail", &Schedule::seperateTail)
        .def("as_matmul", &Schedule::asMatMul)
        .def("auto_schedule", &Schedule::autoSchedule)
        .def("auto_fuse", &Schedule::autoFuse)
        .def("auto_parallelize", &Schedule::autoParallelize)
        .def("auto_set_mem_type", &Schedule::autoSetMemType)
        .def("auto_unroll", &Schedule::autoUnroll);
}

} // namespace ir
