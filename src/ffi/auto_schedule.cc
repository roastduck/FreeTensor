#include <auto_schedule/auto_schedule.h>
#include <driver/array.h>
#include <ffi.h>
#include <schedule.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_auto_schedule(py::module_ &m) {
    py::class_<Sketch>(m, "Sketch")
        .def("get_annotation", &Sketch::getAnnotation);
    py::class_<AutoSchedule>(m, "AutoSchedule")
        .def(py::init<const Schedule &, const Ref<Target> &, const Device &,
                      size_t, size_t>())
        .def("n_candidates", &AutoSchedule::nCandidates)
        .def("n_predict", &AutoSchedule::nPredict)
        .def("set_params", &AutoSchedule::setParams, "args"_a,
             "kws"_a = std::unordered_map<std::string, Ref<Array>>())
        .def("search_one_round", &AutoSchedule::SearchOneRound, "n"_a)
        .def("gen_schedules", &AutoSchedule::genSchedules, "sketches"_a)
        .def("gen_features", &AutoSchedule::genFeatures, "schedules"_a)
        .def("test_and_add", &AutoSchedule::testAndAdd, "sketches"_a,
             "schedules"_a)
        .def("get_best_schedule", &AutoSchedule::getBestSchedule);
}

} // namespace ir
