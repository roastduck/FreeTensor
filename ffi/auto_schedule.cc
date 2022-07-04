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
        .def(py::init<
                 const Schedule &, const Ref<Target> &, const Ref<Device> &,
                 const std::function<AutoSchedule::Predicts(
                     const AutoSchedule::Features &)> &,
                 const std::function<void(const AutoSchedule::Features &,
                                          const AutoSchedule::Predicts &)> &,
                 std::string, int,
                 const std::optional<std::unordered_set<std::string>> &, int>(),
             "schedule"_a, "target"_a, "device"_a, "predict_func"_a,
             "update_func"_a, "tag"_a = "", "min_block_size"_a = 0,
             "rule_set"_a, "verbose"_a = 0)
        .def("set_params", &AutoSchedule::setParams, "args"_a,
             "kws"_a = std::unordered_map<std::string, Ref<Array>>())
        .def("search_one_round", &AutoSchedule::searchOneRound, "n"_a,
             "n_exploit"_a, "n_explore"_a)
        .def("gen_features", &AutoSchedule::genFeatures, "schedules"_a)
        .def("test_and_add", &AutoSchedule::testAndAdd, "sketches"_a)
        .def("get_best_schedule", &AutoSchedule::getBestSchedule)
        .def("test_round", &AutoSchedule::testRound,
             "nth_sketch"_a = std::unordered_map<std::string, int>())
        .def("get_flop", &AutoSchedule::getFlop)
        .def("get_tag", &AutoSchedule::getTag)
        .def("get_best_time", &AutoSchedule::getBestTime);
}

} // namespace freetensor
