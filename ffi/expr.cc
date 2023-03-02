#include <expr.h>
#include <ffi.h>
#include <frontend/frontend_var.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_ast_expr(py::module_ &m) {
    auto pyExpr = m.attr("Expr").cast<py::class_<ExprNode, Expr>>();

    py::class_<VarNode, Var>(m, "Var", pyExpr)
        .def_readonly("name", &VarNode::name_);
    py::class_<LoadNode, Load>(m, "Load", pyExpr)
        .def_readonly("var", &LoadNode::var_)
        .def_property_readonly(
            "indices",
            [](const Load &op) -> std::vector<Expr> { return op->indices_; });
    py::class_<IntConstNode, IntConst>(m, "IntConst", pyExpr)
        .def_readonly("val", &IntConstNode::val_);
    py::class_<FloatConstNode, FloatConst>(m, "FloatConst", pyExpr)
        .def_readonly("val", &FloatConstNode::val_);
    py::class_<BoolConstNode, BoolConst>(m, "BoolConst", pyExpr)
        .def_readonly("val", &BoolConstNode::val_);
    py::class_<AddNode, Add>(m, "Add", pyExpr)
        .def_property_readonly("lhs",
                               [](const Add &op) -> Expr { return op->lhs_; })
        .def_property_readonly("rhs",
                               [](const Add &op) -> Expr { return op->rhs_; });
    py::class_<SubNode, Sub>(m, "Sub", pyExpr)
        .def_property_readonly("lhs",
                               [](const Sub &op) -> Expr { return op->lhs_; })
        .def_property_readonly("rhs",
                               [](const Sub &op) -> Expr { return op->rhs_; });
    py::class_<MulNode, Mul>(m, "Mul", pyExpr)
        .def_property_readonly("lhs",
                               [](const Mul &op) -> Expr { return op->lhs_; })
        .def_property_readonly("rhs",
                               [](const Mul &op) -> Expr { return op->rhs_; });
    py::class_<RealDivNode, RealDiv>(m, "RealDiv", pyExpr)
        .def_property_readonly(
            "lhs", [](const RealDiv &op) -> Expr { return op->lhs_; })
        .def_property_readonly(
            "rhs", [](const RealDiv &op) -> Expr { return op->rhs_; });
    py::class_<FloorDivNode, FloorDiv>(m, "FloorDiv", pyExpr)
        .def_property_readonly(
            "lhs", [](const FloorDiv &op) -> Expr { return op->lhs_; })
        .def_property_readonly(
            "rhs", [](const FloorDiv &op) -> Expr { return op->rhs_; });
    py::class_<CeilDivNode, CeilDiv>(m, "CeilDiv", pyExpr)
        .def_property_readonly(
            "lhs", [](const CeilDiv &op) -> Expr { return op->lhs_; })
        .def_property_readonly(
            "rhs", [](const CeilDiv &op) -> Expr { return op->rhs_; });
    py::class_<RoundTowards0DivNode, RoundTowards0Div>(m, "RoundTowards0Div",
                                                       pyExpr)
        .def_property_readonly(
            "lhs", [](const RoundTowards0Div &op) -> Expr { return op->lhs_; })
        .def_property_readonly(
            "rhs", [](const RoundTowards0Div &op) -> Expr { return op->rhs_; });
    py::class_<ModNode, Mod>(m, "Mod", pyExpr)
        .def_property_readonly("lhs",
                               [](const Mod &op) -> Expr { return op->lhs_; })
        .def_property_readonly("rhs",
                               [](const Mod &op) -> Expr { return op->rhs_; });
    py::class_<RemainderNode, Remainder>(m, "Remainder", pyExpr)
        .def_property_readonly(
            "lhs", [](const Remainder &op) -> Expr { return op->lhs_; })
        .def_property_readonly(
            "rhs", [](const Remainder &op) -> Expr { return op->rhs_; });
    py::class_<MinNode, Min>(m, "Min", pyExpr)
        .def_property_readonly("lhs",
                               [](const Min &op) -> Expr { return op->lhs_; })
        .def_property_readonly("rhs",
                               [](const Min &op) -> Expr { return op->rhs_; });
    py::class_<MaxNode, Max>(m, "Max", pyExpr)
        .def_property_readonly("lhs",
                               [](const Max &op) -> Expr { return op->lhs_; })
        .def_property_readonly("rhs",
                               [](const Max &op) -> Expr { return op->rhs_; });
    py::class_<LTNode, LT>(m, "LT", pyExpr)
        .def_property_readonly("lhs",
                               [](const LT &op) -> Expr { return op->lhs_; })
        .def_property_readonly("rhs",
                               [](const LT &op) -> Expr { return op->rhs_; });
    py::class_<LENode, LE>(m, "LE", pyExpr)
        .def_property_readonly("lhs",
                               [](const LE &op) -> Expr { return op->lhs_; })
        .def_property_readonly("rhs",
                               [](const LE &op) -> Expr { return op->rhs_; });
    py::class_<GTNode, GT>(m, "GT", pyExpr)
        .def_property_readonly("lhs",
                               [](const GT &op) -> Expr { return op->lhs_; })
        .def_property_readonly("rhs",
                               [](const GT &op) -> Expr { return op->rhs_; });
    py::class_<GENode, GE>(m, "GE", pyExpr)
        .def_property_readonly("lhs",
                               [](const GE &op) -> Expr { return op->lhs_; })
        .def_property_readonly("rhs",
                               [](const GE &op) -> Expr { return op->rhs_; });
    py::class_<EQNode, EQ>(m, "EQ", pyExpr)
        .def_property_readonly("lhs",
                               [](const EQ &op) -> Expr { return op->lhs_; })
        .def_property_readonly("rhs",
                               [](const EQ &op) -> Expr { return op->rhs_; });
    py::class_<NENode, NE>(m, "NE", pyExpr)
        .def_property_readonly("lhs",
                               [](const NE &op) -> Expr { return op->lhs_; })
        .def_property_readonly("rhs",
                               [](const NE &op) -> Expr { return op->rhs_; });
    py::class_<LAndNode, LAnd>(m, "LAnd", pyExpr)
        .def_property_readonly("lhs",
                               [](const LAnd &op) -> Expr { return op->lhs_; })
        .def_property_readonly("rhs",
                               [](const LAnd &op) -> Expr { return op->rhs_; });
    py::class_<LOrNode, LOr>(m, "LOr", pyExpr)
        .def_property_readonly("lhs",
                               [](const LOr &op) -> Expr { return op->lhs_; })
        .def_property_readonly("rhs",
                               [](const LOr &op) -> Expr { return op->rhs_; });
    py::class_<LNotNode, LNot>(m, "LNot", pyExpr)
        .def_property_readonly(
            "expr", [](const LNot &op) -> Expr { return op->expr_; });
    py::class_<SqrtNode, Sqrt>(m, "Sqrt", pyExpr)
        .def_property_readonly(
            "expr", [](const Sqrt &op) -> Expr { return op->expr_; });
    py::class_<ExpNode, Exp>(m, "Exp", pyExpr)
        .def_property_readonly("expr",
                               [](const Exp &op) -> Expr { return op->expr_; });
    py::class_<SquareNode, Square>(m, "Square", pyExpr)
        .def_property_readonly(
            "expr", [](const Square &op) -> Expr { return op->expr_; });
    py::class_<SigmoidNode, Sigmoid>(m, "Sigmoid", pyExpr)
        .def_property_readonly(
            "expr", [](const Sigmoid &op) -> Expr { return op->expr_; });
    py::class_<TanhNode, Tanh>(m, "Tanh", pyExpr)
        .def_property_readonly(
            "expr", [](const Tanh &op) -> Expr { return op->expr_; });
    py::class_<AbsNode, Abs>(m, "Abs", pyExpr)
        .def_property_readonly("expr",
                               [](const Abs &op) -> Expr { return op->expr_; });
    py::class_<FloorNode, Floor>(m, "Floor", pyExpr)
        .def_property_readonly(
            "expr", [](const Floor &op) -> Expr { return op->expr_; });
    py::class_<CeilNode, Ceil>(m, "Ceil", pyExpr)
        .def_property_readonly(
            "expr", [](const Ceil &op) -> Expr { return op->expr_; });
    py::class_<IfExprNode, IfExpr>(m, "IfExpr", pyExpr)
        .def_property_readonly(
            "cond", [](const IfExpr &op) -> Expr { return op->cond_; })
        .def_property_readonly(
            "then_case", [](const IfExpr &op) -> Expr { return op->thenCase_; })
        .def_property_readonly("else_case", [](const IfExpr &op) -> Expr {
            return op->elseCase_;
        });
    py::class_<CastNode, Cast>(m, "Cast", pyExpr)
        .def_property_readonly("expr",
                               [](const Cast &op) -> Expr { return op->expr_; })
        .def_property_readonly("dest_type", [](const Cast &op) -> DataType {
            return op->destType_;
        });
    py::class_<IntrinsicNode, Intrinsic> pyIntrinsic(m, "Intrinsic", pyExpr);
    py::class_<AnyExprNode, AnyExpr> pyAnyExpr(m, "AnyExpr", pyExpr);
    py::class_<LoadAtVersionNode, LoadAtVersion>(m, "LoadAtVersion", pyExpr)
        .def_readonly("tape_name", &LoadAtVersionNode::tapeName_)
        .def_property_readonly(
            "indices", [](const LoadAtVersion &op) -> std::vector<Expr> {
                return op->indices_;
            });

    // NOTE: ORDER of the constructor matters!
    pyExpr.def(py::init([](const Expr &expr) { return deepCopy(expr); }))
        .def(py::init([](bool val) { return makeBoolConst(val); }))
        .def(py::init([](int64_t val) { return makeIntConst(val); }))
        .def(py::init([](float val) { return makeFloatConst(val); }))
        .def(py::init([](const FrontendVar &var) { return var.asLoad(); }))
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
            [](const Expr &lhs, const Expr &rhs) {
                return makeRealDiv(lhs, rhs);
            },
            py::is_operator())
        .def(
            "__rtruediv__",
            [](const Expr &rhs, const Expr &lhs) {
                return makeRealDiv(lhs, rhs);
            },
            py::is_operator())
        .def(
            "__floordiv__",
            [](const Expr &lhs, const Expr &rhs) {
                return makeFloorDiv(lhs, rhs);
            },
            py::is_operator())
        .def(
            "__rfloordiv__",
            [](const Expr &rhs, const Expr &lhs) {
                return makeFloorDiv(lhs, rhs);
            },
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
        .def(
            "__neg__",
            [](const Expr &expr) { return makeSub(makeIntConst(0), expr); },
            py::is_operator())
        .def_property_readonly(
            "dtype", [](const Expr &op) -> DataType { return op->dtype(); });
    py::implicitly_convertible<int, ExprNode>();
    py::implicitly_convertible<float, ExprNode>();
    py::implicitly_convertible<bool, ExprNode>();
    py::implicitly_convertible<FrontendVar, ExprNode>();

    // makers
    m.def("makeAnyExpr", &_makeAnyExpr);
    m.def("makeVar", &_makeVar, "name"_a);
    m.def("makeIntConst", &_makeIntConst, "val"_a);
    m.def("makeFloatConst", &_makeFloatConst, "val"_a);
    m.def("makeBoolConst", &_makeBoolConst, "val"_a);
    m.def("makeLoad",
          static_cast<Expr (*)(const std::string &, const std::vector<Expr> &,
                               DataType)>(&_makeLoad),
          "var"_a, "indices"_a, "load_type"_a);
    m.def("makeRemainder",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&_makeRemainder),
          "lhs"_a, "rhs"_a);
    m.def("makeMin",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&_makeMin), "lhs"_a,
          "rhs"_a);
    m.def("makeMax",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&_makeMax), "lhs"_a,
          "rhs"_a);
    m.def("makeLAnd",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&_makeLAnd),
          "lhs"_a, "rhs"_a);
    m.def("makeLOr",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&_makeLOr), "lhs"_a,
          "rhs"_a);
    m.def("makeLNot", static_cast<Expr (*)(const Expr &)>(&_makeLNot),
          "expr"_a);
    m.def("makeSqrt", static_cast<Expr (*)(const Expr &)>(&_makeSqrt),
          "expr"_a);
    m.def("makeExp", static_cast<Expr (*)(const Expr &)>(&_makeExp), "expr"_a);
    m.def("makeSquare", static_cast<Expr (*)(const Expr &)>(&_makeSquare),
          "expr"_a);
    m.def("makeSigmoid", static_cast<Expr (*)(const Expr &)>(&_makeSigmoid),
          "expr"_a);
    m.def("makeTanh", static_cast<Expr (*)(const Expr &)>(&_makeTanh),
          "expr"_a);
    m.def("makeAbs", static_cast<Expr (*)(const Expr &)>(&_makeAbs), "expr"_a);
    m.def("makeFloor", static_cast<Expr (*)(const Expr &)>(&_makeFloor),
          "expr"_a);
    m.def("makeCeil", static_cast<Expr (*)(const Expr &)>(&_makeCeil),
          "expr"_a);
    m.def("makeIfExpr",
          static_cast<Expr (*)(const Expr &, const Expr &, const Expr &)>(
              &_makeIfExpr),
          "cond"_a, "thenCase"_a, "elseCase"_a);
    m.def("makeCast", static_cast<Expr (*)(const Expr &, DataType)>(&_makeCast),
          "expr"_a, "dtype"_a);
    m.def("makeFloorDiv",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&_makeFloorDiv),
          "lhs"_a, "rhs"_a);
    m.def("makeCeilDiv",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&_makeCeilDiv),
          "lhs"_a, "rhs"_a);
    m.def("makeRoundTowards0Div",
          static_cast<Expr (*)(const Expr &, const Expr &)>(
              &_makeRoundTowards0Div),
          "lhs"_a, "rhs"_a);
    m.def("makeIntrinsic",
          static_cast<Expr (*)(const std::string &, const std::vector<Expr> &,
                               DataType, bool)>(&_makeIntrinsic),
          "fmt"_a, "params"_a, "retType"_a = DataType::Void,
          "hasSideEffect"_a = false);
    m.def("makeLoadAtVersion",
          static_cast<Expr (*)(const std::string &, const std::vector<Expr> &,
                               DataType)>(&_makeLoadAtVersion),
          "tape_name"_a, "indices"_a, "load_type"_a);
}

} // namespace freetensor
