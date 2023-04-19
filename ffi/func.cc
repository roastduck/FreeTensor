#include <ffi.h>
#include <func.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_ast_func(py::module_ &m) {
    py::class_<FuncParam>(m, "FuncParam")
        .def(py::init<FuncParam>())
        .def(py::init([](const std::string &name, const Ref<Array> &closure,
                         bool updateClosure) {
                 if (closure.isValid()) {
                     // Because we load Array back to C++ part, we cannot keep
                     // tracking user data with py::keep_alive
                     closure->makePrivateCopy();
                 }
                 return FuncParam{name,
                                  closure.isValid()
                                      ? Ref<Ref<Array>>::make(closure)
                                      : nullptr,
                                  updateClosure};
             }),
             "name"_a, "closure"_a = nullptr, "update_closure"_a = false)
        .def_readonly("name", &FuncParam::name_)
        .def_readonly("update_closure", &FuncParam::updateClosure_)
        .def_property_readonly("is_in_closure", &FuncParam::isInClosure);
    py::implicitly_convertible<std::string, FuncParam>();

    auto initFuncRet = [](const std::string &name, DataType dtype,
                          const Ref<Array> &closure, bool returnClosure) {
        if (closure.isValid()) {
            // Because we load Array back to C++ part, we cannot keep
            // tracking user data with py::keep_alive
            closure->makePrivateCopy();
        }
        return FuncRet{name, dtype,
                       closure.isValid() ? Ref<Ref<Array>>::make(closure)
                                         : nullptr,
                       returnClosure};
    };
    py::class_<FuncRet>(m, "FuncRet")
        .def(py::init<FuncRet>())
        .def(py::init(initFuncRet), "name"_a, "dtype"_a, "closure"_a = nullptr,
             "return_closure"_a = false)
        .def(py::init([&](const std::string &name, const std::string &dtype,
                          const Ref<Array> &closure, bool returnClosure) {
                 return initFuncRet(name, parseDType(dtype), closure,
                                    returnClosure);
             }),
             "name"_a, "dtype"_a, "closure"_a = nullptr,
             "return_closure"_a = false)
        .def_readonly("name", &FuncRet::name_)
        .def_readonly("dtype", &FuncRet::dtype_)
        .def_readonly("return_closure", &FuncRet::returnClosure_)
        .def_property_readonly("is_in_closure", &FuncRet::isInClosure);

    m.attr("Func")
        .cast<py::class_<FuncNode, Func>>()
        .def(py::init([](const std::string &name,
                         const std::vector<FuncParam> &_params,
                         const std::vector<FuncRet> &_returns, const Stmt &body,
                         const std::unordered_map<std::string, Ref<Array>>
                             &extraClosure) {
                 std::vector<FuncParam> params = _params;
                 std::vector<FuncRet> returns = _returns;
                 for (auto &p : params) {
                     if (auto it = extraClosure.find(p.name_);
                         it != extraClosure.end()) {
                         p.closure_ = Ref<Ref<Array>>::make(it->second);
                     }
                 }
                 for (auto &r : returns) {
                     if (auto it = extraClosure.find(r.name_);
                         it != extraClosure.end()) {
                         r.closure_ = Ref<Ref<Array>>::make(it->second);
                     }
                 }
                 return makeFunc(name, std::move(params), std::move(returns),
                                 body);
             }),
             "name"_a, "params"_a, "returns"_a, "body"_a,
             "extra_closure"_a = std::unordered_map<std::string, Ref<Array>>{})
        .def_readonly("name", &FuncNode::name_)
        .def_readonly("params", &FuncNode::params_)
        .def_readonly("returns", &FuncNode::returns_)
        .def_property_readonly(
            "body", [](const Func &op) -> Stmt { return op->body_; });
}

} // namespace freetensor
