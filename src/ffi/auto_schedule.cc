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
        .def(py::init<const Schedule &, const Ref<Target> &, const Device &>())
        .def("set_params", &AutoSchedule::setParams, "args"_a,
             "kws"_a = std::unordered_map<std::string, Array *>())
        .def("init", &AutoSchedule::init, "n_candidates"_a)
        .def("get_random_sketches", &AutoSchedule::getRandomSketches, "n"_a)
        .def("test_and_add", &AutoSchedule::testAndAdd, "sketches"_a)
        .def("get_best_schedule", &AutoSchedule::getBestSchedule);
}

} // namespace ir
