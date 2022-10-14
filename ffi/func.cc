#include <ffi.h>
#include <func.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_ast_func(py::module_ &m) {
    py::class_<FuncParam>(m, "FuncParam")
        .def_readonly("name", &FuncParam::name_)
        .def_readonly("update_closure", &FuncParam::updateClosure_)
        .def_property_readonly("is_in_closure", &FuncParam::isInClosure);

    py::class_<FuncRet>(m, "FuncRet")
        .def_readonly("name", &FuncRet::name_)
        .def_readonly("dtype", &FuncRet::dtype_)
        .def_readonly("return_closure", &FuncRet::returnClosure_)
        .def_property_readonly("is_in_closure", &FuncRet::isInClosure);

    m.attr("Func")
        .cast<py::class_<FuncNode, Func>>()
        .def(
            py::init(
                [](const std::string &name,
                   const std::vector<std::string> &_params,
                   const std::vector<std::pair<std::string, DataType>>
                       &_returns,
                   const Stmt &body,
                   const std::unordered_map<std::string, Ref<Array>> &closure) {
                    std::vector<FuncParam> params;
                    std::vector<FuncRet> returns;
                    params.reserve(_params.size());
                    returns.reserve(_returns.size());
                    for (auto &&p : _params) {
                        if (closure.count(p)) {
                            auto arr = closure.at(p);
                            arr->makePrivateCopy(); // Because we load Array
                                                    // back to C++ part, we
                                                    // cannot keep tracking user
                                                    // data with py::keep_alive
                            params.emplace_back(p, Ref<Ref<Array>>::make(arr),
                                                false);
                        } else {
                            params.emplace_back(p, nullptr, false);
                        }
                    }
                    for (auto &&[p, dtype] : _returns) {
                        if (closure.count(p)) {
                            auto arr = closure.at(p);
                            arr->makePrivateCopy(); // Because we load Array
                                                    // back to C++ part, we
                                                    // cannot keep tracking user
                                                    // data with py::keep_alive
                            returns.emplace_back(
                                p, dtype, Ref<Ref<Array>>::make(arr), false);
                        } else {
                            returns.emplace_back(p, dtype, nullptr, false);
                        }
                    }
                    return makeFunc(name, std::move(params), std::move(returns),
                                    body);
                }),
            "name"_a, "params"_a, "returns"_a, "body"_a, "closure"_a)
        .def_readonly("name", &FuncNode::name_)
        .def_readonly("params", &FuncNode::params_)
        .def_readonly("returns", &FuncNode::returns_)
        .def_property_readonly(
            "body", [](const Func &op) -> Stmt { return op->body_; });
}

} // namespace freetensor
