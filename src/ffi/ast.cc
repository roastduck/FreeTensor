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
    py::class_<Expr>(m, "Expr").def(
        "__repr__", [](const Expr &op) { return printPass(op); });
    py::implicitly_convertible<Stmt, AST>();
    py::implicitly_convertible<Expr, AST>();

    m.def("makeStmtSeq", &makeStmtSeq);
    m.def("makeVarDef", &makeVarDef);
    m.def("makeVar", &makeVar);
    m.def("makeStore", &makeStore);
    m.def("makeLoad", &makeLoad);
    m.def("makeIntConst", &makeIntConst);
    m.def("makeFloatConst", &makeFloatConst);
}

} // namespace ir

