#include <ffi.h>
#include <for_property.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_for_property(py::module_ &m) {
    py::class_<ForProperty, Ref<ForProperty>>(m, "ForProperty")
        .def(py::init<>())
        .def_readonly("parallel", &ForProperty::parallel_)
        .def_readonly("unroll", &ForProperty::unroll_)
        .def_readonly("vectorize", &ForProperty::vectorize_)
        .def_readonly("no_deps", &ForProperty::noDeps_)
        .def_readonly("prefer_libs", &ForProperty::preferLibs_)
        .def("with_parallel", &ForProperty::withParallel, "parallel"_a)
        .def("with_unroll", &ForProperty::withUnroll, "unroll"_a = true)
        .def("with_vectorize", &ForProperty::withVectorize,
             "vectorize"_a = true)
        .def("with_no_deps", &ForProperty::withNoDeps, "var"_a)
        .def("with_prefer_libs", &ForProperty::withPreferLibs, "prefer_libs"_a);
}

} // namespace ir
