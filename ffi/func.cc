#include <ffi.h>
#include <func.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_ast_func(py::module_ &m) {
    m.attr("Func")
        .cast<py::class_<FuncNode, Func>>()
        .def_readonly("name", &FuncNode::name_)
        .def_readonly("params", &FuncNode::params_)
        .def_readonly("returns", &FuncNode::returns_)
        .def_property_readonly(
            "body", [](const Func &op) -> Stmt { return op->body_; });

    // maker
    m.def(
        "makeFunc",
        [](const std::string &name, const std::vector<std::string> &params,
           const std::vector<std::pair<std::string, DataType>> &returns,
           const Stmt &body,
           const std::unordered_map<std::string, Ref<Array>> &_closure) {
            std::unordered_map<std::string, Ref<Ref<Array>>> closure;
            for (auto &&[name, var] : _closure) {
                closure[name] = Ref<Ref<Array>>::make(var);
            }
            return makeFunc(name, params, returns, body, closure);
        },
        "name"_a, "params"_a, "returns"_a, "body"_a, "closure"_a);
}

} // namespace freetensor
