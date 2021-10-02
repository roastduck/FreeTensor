#include <auto_schedule/auto_schedule.h>
#include <auto_schedule/rule.h>
#include <driver/array.h>
#include <ffi.h>
#include <memory>
#include <schedule.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_auto_schedule(py::module_ &m) {
    py::class_<Sketch>(m, "Sketch")
        .def("get_annotation", &Sketch::getAnnotation);
    py::class_<Rule, Ref<Rule>>(m, "Rule");
    py::class_<AutoSchedule>(m, "AutoSchedule")
        .def(py::init<const Schedule &, const Ref<Target> &, const Device &,
                      size_t, size_t>())
        .def("n_candidates", &AutoSchedule::nCandidates)
        .def("n_predict", &AutoSchedule::nPredict)
        .def("set_params", &AutoSchedule::setParams, "args"_a,
             "kws"_a = std::unordered_map<std::string, Array *>())
        .def("get_random_sketches", &AutoSchedule::getRandomSketches, "n"_a)
        .def("gen_schedules", &AutoSchedule::genSchedules, "sketches"_a)
        .def("gen_features", &AutoSchedule::genFeatures, "schedules"_a)
        .def("test_and_add", &AutoSchedule::testAndAdd, "sketches"_a,
             "schedules"_a)
        .def("get_best_schedule", &AutoSchedule::getBestSchedule)
        .def("get_rule_num", &AutoSchedule::getRulesNum)
        .def("rule_analyse", &AutoSchedule::ruleAnalyse, "idx"_a)
        .def("rule_apply", &AutoSchedule::ruleApply, "idx"_a, "sketchPartIdx"_a)
        .def("get_rule", &AutoSchedule::getRule, "idx"_a)
        .def("set_rule", &AutoSchedule::setRule, "idx"_a, "rule"_a)
        .def("get_best_time", &AutoSchedule::getBestTime)
        .def("get_current", &AutoSchedule::getCurrent)
        .def("set_current", &AutoSchedule::setCurrent, "current"_a)
        .def("get_base_sketch", &AutoSchedule::getBaseSketch)
        .def("set_base_sketch", &AutoSchedule::setBaseSketch, "baseSketch"_a);
}

} // namespace ir
