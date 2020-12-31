#include <debug.h>
#include <expr.h>
#include <ffi.h>
#include <stmt.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_ast(py::module_ &m) {
    py::class_<AST>(m, "AST")
        .def(py::init<>())
        .def(py::init<const Stmt &>())
        .def(py::init<const Expr &>())
        .def("match",
             [](const AST &op, const AST &other) { return match(op, other); })
        .def("__str__",
             [](const AST &op) { return op.isValid() ? toString(op) : ""; })
        .def("__repr__", [](const AST &op) {
            return op.isValid() ? "<AST: " + toString(op) + ">" : "None";
        });
    py::class_<Stmt>(m, "Stmt")
        .def(py::init<>())
        .def("match",
             [](const AST &op, const AST &other) { return match(op, other); })
        .def("__str__",
             [](const Stmt &op) { return op.isValid() ? toString(op) : ""; })
        .def("__repr__", [](const Stmt &op) {
            return op.isValid() ? "<Stmt: " + toString(op) + ">" : "None";
        });
    py::class_<Expr>(m, "Expr")
        .def(py::init<>())
        .def(py::init([](int val) { return makeIntConst(val); }))
        .def(py::init([](float val) { return makeFloatConst(val); }))
        .def("match",
             [](const AST &op, const AST &other) { return match(op, other); })
        .def(
            "__add__",
            [](const Expr &lhs, const Expr &rhs) { return makeAdd(lhs, rhs); },
            py::is_operator())
        .def(
            "__radd__",
            [](const Expr &rhs, const Expr &lhs) { return makeAdd(lhs, rhs); },
            py::is_operator())
        .def(
            "__sub__",
            [](const Expr &lhs, const Expr &rhs) { return makeSub(lhs, rhs); },
            py::is_operator())
        .def(
            "__rsub__",
            [](const Expr &rhs, const Expr &lhs) { return makeSub(lhs, rhs); },
            py::is_operator())
        .def(
            "__mul__",
            [](const Expr &lhs, const Expr &rhs) { return makeMul(lhs, rhs); },
            py::is_operator())
        .def(
            "__rmul__",
            [](const Expr &rhs, const Expr &lhs) { return makeMul(lhs, rhs); },
            py::is_operator())
        .def(
            "__truediv__",
            [](const Expr &lhs, const Expr &rhs) { return makeDiv(lhs, rhs); },
            py::is_operator())
        .def(
            "__rtruediv__",
            [](const Expr &rhs, const Expr &lhs) { return makeDiv(lhs, rhs); },
            py::is_operator())
        .def(
            "__floorDiv__",
            [](const Expr &lhs, const Expr &rhs) { return makeDiv(lhs, rhs); },
            py::is_operator())
        .def(
            "__rfloorDiv__",
            [](const Expr &rhs, const Expr &lhs) { return makeDiv(lhs, rhs); },
            py::is_operator())
        .def(
            "__mod__",
            [](const Expr &lhs, const Expr &rhs) { return makeMod(lhs, rhs); },
            py::is_operator())
        .def(
            "__rmod__",
            [](const Expr &rhs, const Expr &lhs) { return makeMod(lhs, rhs); },
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
             [](const Expr &op) { return op.isValid() ? toString(op) : ""; })
        .def("__repr__", [](const Expr &op) {
            return op.isValid() ? "<Expr: " + toString(op) + ">" : "None";
        });
    py::implicitly_convertible<Stmt, AST>();
    py::implicitly_convertible<Expr, AST>();
    py::implicitly_convertible<int, Expr>();
    py::implicitly_convertible<float, Expr>();

    m.def("makeAny", &makeAny);
    m.def("makeStmtSeq",
          static_cast<Stmt (*)(const std::string &, const std::vector<Stmt> &)>(
              &makeStmtSeq),
          "id"_a, "stmts"_a);
    m.def("makeVarDef",
          static_cast<Stmt (*)(const std::string &, const std::string &,
                               const Buffer &, const Stmt &)>(&makeVarDef),
          "nid"_a, "name"_a, "buffer"_a, "body"_a);
    m.def("makeVar", &makeVar, "name"_a);
    m.def("makeStore",
          static_cast<Stmt (*)(const std::string &, const std::string &,
                               const std::vector<Expr> &, const Expr &)>(
              &makeStore),
          "nid"_a, "var"_a, "indices"_a, "expr"_a);
    m.def("makeLoad", &makeLoad, "var"_a, "indices"_a);
    m.def("makeIntConst", &makeIntConst, "val"_a);
    m.def("makeFloatConst", &makeFloatConst, "val"_a);
    m.def("makeFor",
          static_cast<Stmt (*)(const std::string &, const std::string &,
                               const Expr &, const Expr &, const Stmt &)>(
              &makeFor),
          "nid"_a, "iter"_a, "begin"_a, "end"_a, "body"_a);
    m.def("makeIf",
          static_cast<Stmt (*)(const std::string &, const Expr &, const Stmt &,
                               const Stmt &)>(&makeIf),
          "nid"_a, "cond"_a, "thenCase"_a, "elseCase"_a = Stmt());
}

} // namespace ir

