#include <ffi.h>
#include <for_property.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_for_property(py::module_ &m) {
    py::class_<ReductionItem, Ref<ReductionItem>>(m, "ReductionItem")
        .def_readonly("op", &ReductionItem::op_)
        .def_readonly("var", &ReductionItem::var_)
        .def_property_readonly(
            "begins",
            [](const Ref<ReductionItem> &ri) -> std::vector<Expr> {
                return ri->begins_;
            })
        .def_property_readonly(
            "ends", [](const Ref<ReductionItem> &ri) -> std::vector<Expr> {
                return ri->ends_;
            });

    py::class_<ForProperty, Ref<ForProperty>>(m, "ForProperty")
        .def(py::init<>())
        .def_readonly("parallel", &ForProperty::parallel_)
        .def_readonly("unroll", &ForProperty::unroll_)
        .def_readonly("vectorize", &ForProperty::vectorize_)
        .def_readonly("no_deps", &ForProperty::noDeps_)
        .def_readonly("prefer_libs", &ForProperty::preferLibs_)
        .def_property_readonly(
            "reductions",
            [](const Ref<ForProperty> &p) -> std::vector<Ref<ReductionItem>> {
                return p->reductions_;
            })
        .def("with_parallel", &ForProperty::withParallel, "parallel"_a)
        .def("with_unroll", &ForProperty::withUnroll, "unroll"_a = true)
        .def("with_vectorize", &ForProperty::withVectorize,
             "vectorize"_a = true)
        .def("with_no_deps", &ForProperty::withNoDeps, "var"_a)
        .def("with_prefer_libs", &ForProperty::withPreferLibs, "prefer_libs"_a);
}

} // namespace freetensor
