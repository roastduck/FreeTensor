#include <ffi.h>
#include <frontend/frontend_var.h>
#include <frontend/inlined_invoke.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_frontend(py::module_ &m) {
    py::class_<FrontendVarIdx>(m, "FrontendVarIdx")
        .def(py::init<FrontendVarIdx>())
        .def(py::init(&FrontendVarIdx::fromSingle), "single"_a)
        .def(py::init(&FrontendVarIdx::fromSlice), "start"_a, "stop"_a,
             "length"_a = nullptr)
        .def("__repr__",
             [](const FrontendVarIdx &idx) { return toString(idx); });

    m.def("all_reads", static_cast<std::unordered_set<std::string> (*)(
                           const FrontendVarIdx &)>(&allReads));

    py::class_<FrontendVar, Ref<FrontendVar>>(m, "FrontendVar")
        .def(py::init<const std::string &, const std::vector<Expr> &, DataType,
                      MemType, const std::vector<FrontendVarIdx> &, bool>())
        .def_property_readonly("name", &FrontendVar::name)
        .def_property_readonly("full_shape", &FrontendVar::fullShape)
        .def_property_readonly("indices", &FrontendVar::indices)
        .def_property_readonly("dtype", &FrontendVar::dtype)
        .def_property_readonly("mtype", &FrontendVar::mtype)
        .def_property_readonly("ndim", &FrontendVar::ndim)
        .def("shape", static_cast<Expr (FrontendVar::*)(const Expr &) const>(
                          &FrontendVar::shape))
        .def("shape", static_cast<std::vector<Expr> (FrontendVar::*)() const>(
                          &FrontendVar::shape))
        .def("as_load", &FrontendVar::asLoad)
        .def("as_store", &FrontendVar::asStore, "metadata"_a, "value"_a)
        .def("as_reduce_to", &FrontendVar::asReduceTo, "op"_a, "metadata"_a,
             "value"_a, "atomic"_a = false)
        .def("chain_indices", &FrontendVar::chainIndices)
        .def("__repr__", [](const FrontendVar &var) { return toString(var); });

    m.def("strip_returns", &stripReturns);
    m.def("inlined_invoke", &inlinedInvoke, "call_site_metadata"_a, "func"_a,
          "args"_a, "kvs"_a, "ret_names"_a, "conflict_names"_a,
          "force_allow_closures"_a = false);
}

} // namespace freetensor
