#include <expr.h>
#include <ffi.h>
#include <pass/print_pass.h>
#include <stmt.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_ast(py::module_ &m) {
    py::class_<AST>(m, "AST")
        .def(py::init<>())
        .def(py::init<const Stmt &>())
        .def(py::init<const Expr &>())
        .def("__str__",
             [](const AST &op) { return op.isValid() ? printPass(op) : ""; })
        .def("__repr__", [](const AST &op) {
            return op.isValid() ? "<AST: " + printPass(op) + ">" : "None";
        });
    py::class_<Stmt>(m, "Stmt")
        .def(py::init<>())
        .def("__str__",
             [](const Stmt &op) { return op.isValid() ? printPass(op) : ""; })
        .def("__repr__", [](const Stmt &op) {
            return op.isValid() ? "<Stmt: " + printPass(op) + ">" : "None";
        });
    py::class_<Expr>(m, "Expr")
        .def(py::init<>())
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
        .def(
            "__lt__",
            [](const Expr &lhs, const Expr &rhs) { return makeLT(lhs, rhs); },
            py::is_operator())
        .def(
            "__le__",
            [](const Expr &lhs, const Expr &rhs) { return makeLE(lhs, rhs); },
            py::is_operator())
        .def(
            "__gt__",
            [](const Expr &lhs, const Expr &rhs) { return makeGT(lhs, rhs); },
            py::is_operator())
        .def(
            "__ge__",
            [](const Expr &lhs, const Expr &rhs) { return makeGE(lhs, rhs); },
            py::is_operator())
        .def(
            "__eq__",
            [](const Expr &lhs, const Expr &rhs) { return makeEQ(lhs, rhs); },
            py::is_operator())
        .def(
            "__ne__",
            [](const Expr &lhs, const Expr &rhs) { return makeNE(lhs, rhs); },
            py::is_operator())
        .def("__str__",
             [](const Expr &op) { return op.isValid() ? printPass(op) : ""; })
        .def("__repr__", [](const Expr &op) {
            return op.isValid() ? "<Expr: " + printPass(op) + ">" : "None";
        });
    py::implicitly_convertible<Stmt, AST>();
    py::implicitly_convertible<Expr, AST>();
    py::implicitly_convertible<int, Expr>();
    py::implicitly_convertible<float, Expr>();

    m.def("makeStmtSeq", &makeStmtSeq, "stmts"_a);
    m.def("makeVarDef", &makeVarDef, "name"_a, "buffer"_a, "body"_a);
    m.def("makeVar", &makeVar, "name"_a);
    m.def("makeStore", &makeStore, "var"_a, "indices"_a, "expr"_a);
    m.def("makeLoad", &makeLoad, "var"_a, "indices"_a);
    m.def("makeIntConst", &makeIntConst, "val"_a);
    m.def("makeFloatConst", &makeFloatConst, "val"_a);
    m.def("makeAdd", &makeAdd, "lhs"_a, "rhs"_a);
    m.def("makeSub", &makeSub, "lhs"_a, "rhs"_a);
    m.def("makeMul", &makeMul, "lhs"_a, "rhs"_a);
    m.def("makeDiv", &makeDiv, "lhs"_a, "rhs"_a);
    m.def("makeMod", &makeMod, "lhs"_a, "rhs"_a);
    m.def("makeLT", &makeLT, "lhs"_a, "rhs"_a);
    m.def("makeLE", &makeLE, "lhs"_a, "rhs"_a);
    m.def("makeGT", &makeGT, "lhs"_a, "rhs"_a);
    m.def("makeGE", &makeGE, "lhs"_a, "rhs"_a);
    m.def("makeEQ", &makeEQ, "lhs"_a, "rhs"_a);
    m.def("makeNE", &makeNE, "lhs"_a, "rhs"_a);
    m.def("makeFor", &makeFor, "iter"_a, "begin"_a, "end"_a, "body"_a);
    m.def("makeIf", &makeIf, "cond"_a, "thenCase"_a, "elseCase"_a = Stmt());
}

} // namespace ir

