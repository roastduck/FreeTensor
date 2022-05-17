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
        [](const std::string &name, const std::vector<std::string> &_params,
           const std::vector<std::pair<std::string, DataType>> &_returns,
           const Stmt &body,
           const std::unordered_map<std::string, Ref<Array>> &closure) {
            std::vector<FuncParam> params;
            std::vector<FuncRet> returns;
            params.reserve(_params.size());
            returns.reserve(_returns.size());
            for (auto &&name : _params) {
                params.emplace_back(name,
                                    closure.count(name) ? Ref<Ref<Array>>::make(
                                                              closure.at(name))
                                                        : nullptr,
                                    false);
            }
            for (auto &&[name, dtype] : _returns) {
                returns.emplace_back(
                    name, dtype,
                    closure.count(name)
                        ? Ref<Ref<Array>>::make(closure.at(name))
                        : nullptr,
                    false);
            }
            return makeFunc(name, std::move(params), std::move(returns), body);
        },
        "name"_a, "params"_a, "returns"_a, "body"_a, "closure"_a);
}

} // namespace freetensor
