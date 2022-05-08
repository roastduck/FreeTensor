#include <auto_schedule/auto_schedule.h>
#include <driver/array.h>
#include <ffi.h>
#include <schedule.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_schedule(py::module_ &m) {
    py::enum_<FissionSide>(m, "FissionSide")
        .value("Before", FissionSide::Before)
        .value("After", FissionSide::After);
    py::enum_<MoveToSide>(m, "MoveToSide")
        .value("Before", MoveToSide::Before)
        .value("After", MoveToSide::After);
    py::enum_<VarSplitMode>(m, "VarSplitMode")
        .value("FixedSize", VarSplitMode::FixedSize)
        .value("RelaxedSize", VarSplitMode::RelaxedSize);

    py::class_<Schedule>(m, "Schedule")
        .def(py::init<const Stmt &, int>(), "stmt"_a, "verbose"_a = 0)
        .def(py::init<const Func &, int>(), "func"_a, "verbose"_a = 0)
        .def("ast", &Schedule::ast)
        .def("func", &Schedule::func)
        .def("logs", &Schedule::logs)
        .def("find", static_cast<Stmt (Schedule::*)(
                         const std::function<bool(const Stmt &)> &) const>(
                         &Schedule::find))
        .def("find",
             static_cast<Stmt (Schedule::*)(const ID &) const>(&Schedule::find))
        .def("find_all", static_cast<std::vector<Stmt> (Schedule::*)(
                             const std::function<bool(const Stmt &)> &) const>(
                             &Schedule::findAll))
        .def("find_all",
             static_cast<std::vector<Stmt> (Schedule::*)(const ID &) const>(
                 &Schedule::findAll))
        .def("split", &Schedule::split, "id"_a, "factor"_a = -1,
             "nparts"_a = -1)
        .def("reorder", &Schedule::reorder, "order"_a)
        .def("merge", &Schedule::merge, "loop1"_a, "loop2"_a)
        .def("fission", &Schedule::fission, "loop"_a, "side"_a, "splitter"_a,
             "suffix0"_a = ".a", "suffix1"_a = ".b")
        .def("fuse",
             static_cast<ID (Schedule::*)(const ID &, const ID &, bool)>(
                 &Schedule::fuse),
             "loop0"_a, "loop1"_a, "strict"_a = false)
        .def("fuse",
             static_cast<ID (Schedule::*)(const ID &, bool)>(&Schedule::fuse),
             "loop0"_a, "strict"_a = false)
        .def("swap", &Schedule::swap, "order"_a)
        .def("blend", &Schedule::blend, "loop"_a)
        .def("cache", &Schedule::cache, "stmt"_a, "var"_a, "mtype"_a)
        .def("cache_reduction", &Schedule::cacheReduction, "stmt"_a, "var"_a,
             "mtype"_a)
        .def("set_mem_type", &Schedule::setMemType, "vardef"_a, "mtype"_a)
        .def("var_split", &Schedule::varSplit, "vardef"_a, "dim"_a, "mode"_a,
             "factor"_a = -1, "nparts"_a = -1)
        .def("var_merge", &Schedule::varMerge, "vardef"_a, "dim"_a)
        .def("var_reorder", &Schedule::varReorder, "vardef"_a, "order"_a)
        .def("move_to", &Schedule::moveTo, "stmt"_a, "side"_a, "dst"_a)
        .def("inline", &Schedule::inlining, "vardef"_a)
        .def("parallelize", &Schedule::parallelize, "loop"_a, "parallel"_a)
        .def("unroll", &Schedule::unroll, "loop"_a, "immedate"_a = false)
        .def("vectorize", &Schedule::vectorize, "loop"_a)
        .def("separate_tail", &Schedule::separateTail,
             "noDuplicateVarDefs"_a = false)
        .def("as_matmul", &Schedule::asMatMul)
        .def("auto_schedule", &Schedule::autoSchedule)
        .def("auto_use_lib", &Schedule::autoUseLib)
        .def("auto_fuse", &Schedule::autoFuse)
        .def("auto_parallelize", &Schedule::autoParallelize)
        .def("auto_set_mem_type", &Schedule::autoSetMemType)
        .def("auto_unroll", &Schedule::autoUnroll);
}

} // namespace freetensor
