#include <expr.h>
#include <ffi.h>
#include <pass/print_pass.h>
#include <stmt.h>

namespace ir {

void init_ffi_ast(py::module_ &m) {
    py::class_<AST>(m, "AST")
        .def(py::init<const Stmt &>())
        .def(py::init<const Expr &>())
        .def("__repr__", [](const AST &op) { return printPass(op); });
    py::class_<Stmt>(m, "Stmt").def(
        "__repr__", [](const Stmt &op) { return printPass(op); });
    py::class_<Expr>(m, "Expr")
        .def(py::init([](int val) { return makeIntConst(val); }))
        .def(py::init([](float val) { return makeFloatConst(val); }))
        .def(
            "__add__",
            [](const Expr &lhs, const Expr &rhs) { return makeAdd(lhs, rhs); },
            py::is_operator())
        .def(
            "__sub__",
            [](const Expr &lhs, const Expr &rhs) { return makeSub(lhs, rhs); },
            py::is_operator())
        .def(
            "__mul__",
            [](const Expr &lhs, const Expr &rhs) { return makeMul(lhs, rhs); },
            py::is_operator())
        .def(
            "__truediv__",
            [](const Expr &lhs, const Expr &rhs) { return makeDiv(lhs, rhs); },
            py::is_operator())
        .def(
            "__floorDiv__",
            [](const Expr &lhs, const Expr &rhs) { return makeDiv(lhs, rhs); },
            py::is_operator())
        .def(
            "__mod__",
            [](const Expr &lhs, const Expr &rhs) { return makeMod(lhs, rhs); },
            py::is_operator())
        .def("__repr__", [](const Expr &op) { return printPass(op); });
    py::implicitly_convertible<Stmt, AST>();
    py::implicitly_convertible<Expr, AST>();
    py::implicitly_convertible<int, Expr>();
    py::implicitly_convertible<float, Expr>();

    m.def("makeStmtSeq", &makeStmtSeq);
    m.def("makeVarDef", &makeVarDef);
    m.def("makeVar", &makeVar);
    m.def("makeStore", &makeStore);
    m.def("makeLoad", &makeLoad);
    m.def("makeIntConst", &makeIntConst);
    m.def("makeFloatConst", &makeFloatConst);
    m.def("makeAdd", &makeAdd);
    m.def("makeSub", &makeSub);
    m.def("makeMul", &makeMul);
    m.def("makeDiv", &makeDiv);
    m.def("makeMod", &makeMod);
}

} // namespace ir

