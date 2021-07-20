#include <auto_schedule/auto_schedule.h>
#include <driver/array.h>
#include <ffi.h>
#include <schedule.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_auto_schedule(py::module_ &m) {
    py::class_<Sketch>(m, "Sketch")
        .def("get_annotation", &Sketch::get_annotation);
    py::class_<AutoSchedule>(m, "AutoSchedule")
        .def(py::init<const Schedule &, const Ref<Target> &, const Device &>())
        .def("set_params", &AutoSchedule::set_params, "args"_a,
             "kws"_a = std::unordered_map<std::string, Array *>())
        .def("init", &AutoSchedule::init, "n_candidates"_a)
        .def("get_random_sketches", &AutoSchedule::get_random_sketches, "n"_a)
        .def("test_and_add", &AutoSchedule::test_and_add, "sketches"_a)
        .def("get_best_schedule", &AutoSchedule::get_best_schedule);
}

} // namespace ir
