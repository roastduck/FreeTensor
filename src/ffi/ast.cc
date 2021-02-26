#include <typeinfo>

#include <debug.h>
#include <except.h>
#include <expr.h>
#include <ffi.h>
#include <stmt.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_ast(py::module_ &m) {
    py::class_<ASTNode, AST> pyAST(m, "AST");
    py::class_<StmtNode, Stmt> pyStmt(m, "Stmt", pyAST);
    py::class_<ExprNode, Expr> pyExpr(m, "Expr", pyAST);

    py::class_<StmtSeqNode, StmtSeq> pyStmtSeq(m, "StmtSeq", pyStmt);
    py::class_<VarDefNode, VarDef> pyVarDef(m, "VarDef", pyStmt);
    py::class_<StoreNode, Store> pyStore(m, "Store", pyStmt);
    py::class_<ReduceToNode, ReduceTo> pyReduceTo(m, "ReduceTo", pyStmt);
    py::class_<ForNode, For> pyFor(m, "For", pyStmt);
    py::class_<IfNode, If> pyIf(m, "If", pyStmt);
    py::class_<AssertNode, Assert> pyAssert(m, "Assert", pyStmt);
    py::class_<EvalNode, Eval> pyEval(m, "Eval", pyStmt);
    py::class_<AnyNode, Any> pyAny(m, "Any", pyStmt);

    py::class_<VarNode, Var> pyVar(m, "Var", pyExpr);
    py::class_<LoadNode, Load> pyLoad(m, "Load", pyExpr);
    py::class_<IntConstNode, IntConst> pyIntConst(m, "IntConst", pyExpr);
    py::class_<FloatConstNode, FloatConst> pyFloatConst(m, "FloatConst",
                                                        pyExpr);
    py::class_<AddNode, Add> pyAdd(m, "Add", pyExpr);
    py::class_<SubNode, Sub> pySub(m, "Sub", pyExpr);
    py::class_<MulNode, Mul> pyMul(m, "Mul", pyExpr);
    py::class_<DivNode, Div> pyDiv(m, "Div", pyExpr);
    py::class_<ModNode, Mod> pyMod(m, "Mod", pyExpr);
    py::class_<MinNode, Min> pyMin(m, "Min", pyExpr);
    py::class_<MaxNode, Max> pyMax(m, "Max", pyExpr);
    py::class_<LTNode, LT> pyLT(m, "LT", pyExpr);
    py::class_<LENode, LE> pyLE(m, "LE", pyExpr);
    py::class_<GTNode, GT> pyGT(m, "GT", pyExpr);
    py::class_<GENode, GE> pyGE(m, "GE", pyExpr);
    py::class_<EQNode, EQ> pyEQ(m, "EQ", pyExpr);
    py::class_<NENode, NE> pyNE(m, "NE", pyExpr);
    py::class_<LAndNode, LAnd> pyLAnd(m, "LAnd", pyExpr);
    py::class_<LOrNode, LOr> pyLOr(m, "LOr", pyExpr);
    py::class_<LNotNode, LNot> pyLNot(m, "LNot", pyExpr);
    py::class_<IntrinsicNode, Intrinsic> pyIntrinsic(m, "Intrinsic", pyExpr);

    pyAST
        .def("match",
             [](const AST &op, const AST &other) { return match(op, other); })
        .def("__str__", [](const AST &op) { return toString(op); })
        .def("__repr__", [](const AST &op) {
            return "<" + toString(op->nodeType()) + ": " + toString(op) + ">";
        });

    pyExpr.def(py::init([](int val) { return makeIntConst(val); }))
        .def(py::init([](float val) { return makeFloatConst(val); }))
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
            "__floordiv__",
            [](const Expr &lhs, const Expr &rhs) { return makeDiv(lhs, rhs); },
            py::is_operator())
        .def(
            "__rfloordiv__",
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
            py::is_operator());
    py::implicitly_convertible<int, ExprNode>();
    py::implicitly_convertible<float, ExprNode>();

    // Statements
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
                               const Expr &, const Expr &, const std::string &,
                               const Stmt &)>(&makeFor),
          "nid"_a, "iter"_a, "begin"_a, "end"_a, "parallel"_a, "body"_a);
    m.def("makeIf",
          static_cast<Stmt (*)(const std::string &, const Expr &, const Stmt &,
                               const Stmt &)>(&makeIf),
          "nid"_a, "cond"_a, "thenCase"_a, "elseCase"_a = nullptr);
    m.def("makeEval",
          static_cast<Stmt (*)(const std::string &, const Expr &)>(&makeEval),
          "nid"_a, "expr"_a);

    // Expressions
    m.def("makeMin",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&makeMin), "lhs"_a,
          "rhs"_a);
    m.def("makeMax",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&makeMax), "lhs"_a,
          "rhs"_a);
    m.def("makeLAnd",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&makeLAnd), "lhs"_a,
          "rhs"_a);
    m.def("makeLOr",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&makeLOr), "lhs"_a,
          "rhs"_a);
    m.def("makeLNot", static_cast<Expr (*)(const Expr &)>(&makeLNot), "expr"_a);
    m.def("makeIntrinsic",
          static_cast<Expr (*)(const std::string &, const std::vector<Expr> &)>(
              &makeIntrinsic),
          "fmt"_a, "params"_a);
}

} // namespace ir

namespace pybind11 {

template <> struct polymorphic_type_hook<ir::ASTNode> {
    static const void *get(const ir::ASTNode *src,
                           const std::type_info *&type) {
        if (src == nullptr) {
            return src;
        }
        switch (src->nodeType()) {
#define DISPATCH(name)                                                         \
    case ir::ASTNodeType::name:                                                \
        type = &typeid(ir::name##Node);                                        \
        return static_cast<const ir::name##Node *>(src);

            DISPATCH(StmtSeq);
            DISPATCH(VarDef);
            DISPATCH(Store);
            DISPATCH(ReduceTo);
            DISPATCH(For);
            DISPATCH(If);
            DISPATCH(Assert);
            DISPATCH(Eval);
            DISPATCH(Any);
            DISPATCH(Var);
            DISPATCH(Load);
            DISPATCH(IntConst);
            DISPATCH(FloatConst);
            DISPATCH(Add);
            DISPATCH(Sub);
            DISPATCH(Mul);
            DISPATCH(Div);
            DISPATCH(Mod);
            DISPATCH(Min);
            DISPATCH(Max);
            DISPATCH(LT);
            DISPATCH(LE);
            DISPATCH(GT);
            DISPATCH(GE);
            DISPATCH(EQ);
            DISPATCH(NE);
            DISPATCH(LAnd);
            DISPATCH(LOr);
            DISPATCH(LNot);
            DISPATCH(Intrinsic);
        default:
            ERROR("Unexpected AST node type");
        }
    }
};

template <> struct polymorphic_type_hook<ir::StmtNode> {
    static const void *get(const ir::StmtNode *src,
                           const std::type_info *&type) {
        if (src == nullptr) {
            return src;
        }
        switch (src->nodeType()) {
#define DISPATCH(name)                                                         \
    case ir::ASTNodeType::name:                                                \
        type = &typeid(ir::name##Node);                                        \
        return static_cast<const ir::name##Node *>(src);

            DISPATCH(StmtSeq);
            DISPATCH(VarDef);
            DISPATCH(Store);
            DISPATCH(ReduceTo);
            DISPATCH(For);
            DISPATCH(If);
            DISPATCH(Assert);
            DISPATCH(Eval);
            DISPATCH(Any);
        default:
            ERROR("Unexpected Stmt node type");
        }
    }
};

template <> struct polymorphic_type_hook<ir::ExprNode> {
    static const void *get(const ir::ExprNode *src,
                           const std::type_info *&type) {
        if (src == nullptr) {
            return src;
        }
        switch (src->nodeType()) {
#define DISPATCH(name)                                                         \
    case ir::ASTNodeType::name:                                                \
        type = &typeid(ir::name##Node);                                        \
        return static_cast<const ir::name##Node *>(src);

            DISPATCH(Var);
            DISPATCH(Load);
            DISPATCH(IntConst);
            DISPATCH(FloatConst);
            DISPATCH(Add);
            DISPATCH(Sub);
            DISPATCH(Mul);
            DISPATCH(Div);
            DISPATCH(Mod);
            DISPATCH(Min);
            DISPATCH(Max);
            DISPATCH(LT);
            DISPATCH(LE);
            DISPATCH(GT);
            DISPATCH(GE);
            DISPATCH(EQ);
            DISPATCH(NE);
            DISPATCH(LAnd);
            DISPATCH(LOr);
            DISPATCH(LNot);
            DISPATCH(Intrinsic);
        default:
            ERROR("Unexpected Expr node type");
        }
    }
};

} // namespace pybind11

