#include <auto_schedule/auto_schedule.h>
#include <driver/array.h>
#include <ffi.h>
#include <schedule.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_auto_schedule(py::module_ &m) {
    py::class_<Sketch>(m, "Sketch")
        .def("get_annotation", &Sketch::getAnnotation);
    py::class_<AutoSchedule>(m, "AutoSchedule")
        .def(py::init<const Schedule &, const Ref<Target> &, const Device &,
                      size_t, py::function, py::function>())
        .def(py::init<const Schedule &, const Ref<Target> &, const Device &,
                      size_t, py::function, py::function, std::string>())
        .def("measuredSize", &AutoSchedule::measuredSize)
        .def("set_params", &AutoSchedule::setParams, "args"_a,
             "kws"_a = std::unordered_map<std::string, Ref<Array>>())
        .def("search_one_round", &AutoSchedule::searchOneRound, "n"_a)
        .def("gen_features", &AutoSchedule::genFeatures, "schedules"_a)
        .def("test_and_add", &AutoSchedule::testAndAdd, "sketches"_a)
        .def("get_best_schedule", &AutoSchedule::getBestSchedule)
        .def("test_cache_write", &AutoSchedule::testCacheWrite)
        .def("test_multi_level_tiling_with_fusion",
             &AutoSchedule::testMultiLevelTilingWithFusion, "n_level"_a)
        .def("test_thread_bind", &AutoSchedule::testThreadBind)
        .def("test_cache_read", &AutoSchedule::testCacheRead)
        .def("get_flop", &AutoSchedule::getFlop)
        .def("get_tag", &AutoSchedule::getTag)
        .def("get_best_time", &AutoSchedule::getBestTime);
}

} // namespace freetensor
