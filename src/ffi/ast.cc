#include <typeinfo>

#include <debug.h>
#include <except.h>
#include <expr.h>
#include <ffi.h>
#include <stmt.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_ast(py::module_ &m) {
    py::enum_<ASTNodeType>(m, "ASTNodeType")
        .value("Any", ASTNodeType::Any)
        .value("AnyExpr", ASTNodeType::AnyExpr)
        .value("StmtSeq", ASTNodeType::StmtSeq)
        .value("VarDef", ASTNodeType::VarDef)
        .value("Var", ASTNodeType::Var)
        .value("Store", ASTNodeType::Store)
        .value("Load", ASTNodeType::Load)
        .value("ReduceTo", ASTNodeType::ReduceTo)
        .value("IntConst", ASTNodeType::IntConst)
        .value("FloatConst", ASTNodeType::FloatConst)
        .value("BoolConst", ASTNodeType::BoolConst)
        .value("Add", ASTNodeType::Add)
        .value("Sub", ASTNodeType::Sub)
        .value("Mul", ASTNodeType::Mul)
        .value("RealDiv", ASTNodeType::RealDiv)
        .value("FloorDiv", ASTNodeType::FloorDiv)
        .value("CeilDiv", ASTNodeType::CeilDiv)
        .value("RoundTowards0Div", ASTNodeType::RoundTowards0Div)
        .value("Mod", ASTNodeType::Mod)
        .value("Min", ASTNodeType::Min)
        .value("Max", ASTNodeType::Max)
        .value("LT", ASTNodeType::LT)
        .value("LE", ASTNodeType::LE)
        .value("GT", ASTNodeType::GT)
        .value("GE", ASTNodeType::GE)
        .value("EQ", ASTNodeType::EQ)
        .value("NE", ASTNodeType::NE)
        .value("LAnd", ASTNodeType::LAnd)
        .value("LOr", ASTNodeType::LOr)
        .value("LNot", ASTNodeType::LNot)
        .value("For", ASTNodeType::For)
        .value("If", ASTNodeType::If)
        .value("Assert", ASTNodeType::Assert)
        .value("Intrinsic", ASTNodeType::Intrinsic)
        .value("Eval", ASTNodeType::Eval);

    py::class_<ASTNode, AST> pyAST(m, "AST");
    py::class_<StmtNode, Stmt> pyStmt(m, "Stmt", pyAST);
    py::class_<ExprNode, Expr> pyExpr(m, "Expr", pyAST);

#ifdef IR_DEBUG
    pyAST.def_readonly("debug_creator", &ASTNode::debugCreator_);
#endif

    pyStmt.def_property_readonly("nid", &StmtNode::id);

    py::class_<StmtSeqNode, StmtSeq>(m, "StmtSeq", pyStmt)
        .def_property_readonly("stmts", [](const StmtSeq &op) {
            return std::vector<Stmt>(op->stmts_.begin(), op->stmts_.end());
        });
    py::class_<VarDefNode, VarDef>(m, "VarDef", pyStmt)
        .def_readonly("name", &VarDefNode::name_)
        .def_readonly("buffer", &VarDefNode::buffer_)
        .def_property_readonly(
            "size_lim", [](const VarDef &op) -> Expr { return op->sizeLim_; })
        .def_property_readonly(
            "body", [](const VarDef &op) -> Stmt { return op->body_; });
    py::class_<StoreNode, Store>(m, "Store", pyStmt)
        .def_readonly("var", &StoreNode::var_)
        .def_property_readonly("indices",
                               [](const Store &op) {
                                   return std::vector<Expr>(
                                       op->indices_.begin(),
                                       op->indices_.end());
                               })
        .def_property_readonly(
            "expr", [](const Store &op) -> Expr { return op->expr_; });
    py::class_<ReduceToNode, ReduceTo>(m, "ReduceTo", pyStmt)
        .def_readonly("var", &ReduceToNode::var_)
        .def_property_readonly("indices",
                               [](const ReduceTo &op) {
                                   return std::vector<Expr>(
                                       op->indices_.begin(),
                                       op->indices_.end());
                               })
        .def_readonly("op", &ReduceToNode::op_)
        .def_property_readonly(
            "expr", [](const ReduceTo &op) -> Expr { return op->expr_; });
    py::class_<ForNode, For>(m, "For", pyStmt)
        .def_readonly("iter", &ForNode::iter_)
        .def_property_readonly("begin",
                               [](const For &op) -> Expr { return op->begin_; })
        .def_property_readonly("end",
                               [](const For &op) -> Expr { return op->end_; })
        .def_readonly("parallel", &ForNode::parallel_)
        .def_property_readonly("body",
                               [](const For &op) -> Stmt { return op->body_; });
    py::class_<IfNode, If>(m, "If", pyStmt)
        .def_property_readonly("cond",
                               [](const If &op) -> Expr { return op->cond_; })
        .def_property_readonly(
            "then_case", [](const If &op) -> Stmt { return op->thenCase_; })
        .def_property_readonly(
            "else_case", [](const If &op) -> Stmt { return op->elseCase_; });
    py::class_<AssertNode, Assert>(m, "Assert", pyStmt)
        .def_property_readonly(
            "cond", [](const Assert &op) -> Expr { return op->cond_; })
        .def_property_readonly(
            "body", [](const Assert &op) -> Stmt { return op->body_; });
    py::class_<EvalNode, Eval>(m, "Eval", pyStmt)
        .def_property_readonly(
            "expr", [](const Eval &op) -> Expr { return op->expr_; });
    py::class_<AnyNode, Any> pyAny(m, "Any", pyStmt);

    py::class_<VarNode, Var> pyVar(m, "Var", pyExpr);
    py::class_<LoadNode, Load> pyLoad(m, "Load", pyExpr);
    py::class_<IntConstNode, IntConst> pyIntConst(m, "IntConst", pyExpr);
    py::class_<FloatConstNode, FloatConst> pyFloatConst(m, "FloatConst",
                                                        pyExpr);
    py::class_<BoolConstNode, BoolConst> pyBoolConst(m, "BoolConst", pyExpr);
    py::class_<AddNode, Add> pyAdd(m, "Add", pyExpr);
    py::class_<SubNode, Sub> pySub(m, "Sub", pyExpr);
    py::class_<MulNode, Mul> pyMul(m, "Mul", pyExpr);
    py::class_<RealDivNode, RealDiv> pyRealDiv(m, "RealDiv", pyExpr);
    py::class_<FloorDivNode, FloorDiv> pyFloorDiv(m, "FloorDiv", pyExpr);
    py::class_<CeilDivNode, CeilDiv> pyCeilDiv(m, "CeilDiv", pyExpr);
    py::class_<RoundTowards0DivNode, RoundTowards0Div> pyRoundTowards0Div(
        m, "RoundTowards0Div", pyExpr);
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
    py::class_<AnyExprNode, AnyExpr> pyAnyExpr(m, "AnyExpr", pyExpr);

    pyAST
        .def("match",
             [](const AST &op, const AST &other) { return match(op, other); })
        .def("type", [](const AST &op) { return toString(op->nodeType()); })
        .def("__str__", [](const AST &op) { return toString(op); })
        .def("__repr__", [](const AST &op) {
            return "<" + toString(op->nodeType()) + ": " + toString(op) + ">";
        });

    pyExpr.def(py::init([](int val) { return makeIntConst(val); }))
        .def(py::init([](float val) { return makeFloatConst(val); }))
        .def(py::init([](bool val) { return makeBoolConst(val); }))
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
            py::is_operator());
    py::implicitly_convertible<int, ExprNode>();
    py::implicitly_convertible<float, ExprNode>();
    py::implicitly_convertible<bool, ExprNode>();

    // Statements
    m.def("makeAny", &_makeAny);
    m.def("makeStmtSeq",
          static_cast<Stmt (*)(const std::string &, const std::vector<Stmt> &)>(
              &_makeStmtSeq),
          "id"_a, "stmts"_a);
    m.def(
        "makeVarDef",
        static_cast<Stmt (*)(const std::string &, const std::string &,
                             const Buffer &, const Expr &, const Stmt &, bool)>(
            &_makeVarDef),
        "nid"_a, "name"_a, "buffer"_a, "size_lim"_a, "body"_a, "pinned"_a);
    m.def("makeVar", &_makeVar, "name"_a);
    m.def("makeStore",
          static_cast<Stmt (*)(const std::string &, const std::string &,
                               const std::vector<Expr> &, const Expr &)>(
              &_makeStore),
          "nid"_a, "var"_a, "indices"_a, "expr"_a);
    m.def("makeLoad", &_makeLoad, "var"_a, "indices"_a);
    m.def("makeIntConst", &_makeIntConst, "val"_a);
    m.def("makeFloatConst", &_makeFloatConst, "val"_a);
    m.def("makeFor",
          static_cast<Stmt (*)(const std::string &, const std::string &,
                               const Expr &, const Expr &, const Expr &,
                               const std::string &, const bool, const Stmt &)>(
              &_makeFor),
          "nid"_a, "iter"_a, "begin"_a, "end"_a, "len"_a, "parallel"_a,
          "unroll"_a, "body"_a);
    m.def("makeIf",
          static_cast<Stmt (*)(const std::string &, const Expr &, const Stmt &,
                               const Stmt &)>(&_makeIf),
          "nid"_a, "cond"_a, "thenCase"_a, "elseCase"_a = nullptr);
    m.def(
        "makeAssert",
        static_cast<Stmt (*)(const std::string &, const Expr &, const Stmt &)>(
            &_makeAssert),
        "nid"_a, "cond"_a, "body"_a);
    m.def("makeEval",
          static_cast<Stmt (*)(const std::string &, const Expr &)>(&_makeEval),
          "nid"_a, "expr"_a);

    // Expressions
    m.def("makeAnyExpr", &_makeAnyExpr);
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
    m.def("makeFloorDiv",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&_makeFloorDiv),
          "expr"_a, "expr"_a);
    m.def("makeCeilDiv",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&_makeCeilDiv),
          "expr"_a, "expr"_a);
    m.def("makeRoundTowards0Div",
          static_cast<Expr (*)(const Expr &, const Expr &)>(
              &_makeRoundTowards0Div),
          "expr"_a, "expr"_a);
    m.def("makeIntrinsic",
          static_cast<Expr (*)(const std::string &, const std::vector<Expr> &,
                               DataType)>(&_makeIntrinsic),
          "fmt"_a, "params"_a, "retType"_a = DataType::Void);
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
            DISPATCH(AnyExpr);
            DISPATCH(Var);
            DISPATCH(Load);
            DISPATCH(IntConst);
            DISPATCH(FloatConst);
            DISPATCH(BoolConst);
            DISPATCH(Add);
            DISPATCH(Sub);
            DISPATCH(Mul);
            DISPATCH(RealDiv);
            DISPATCH(FloorDiv);
            DISPATCH(CeilDiv);
            DISPATCH(RoundTowards0Div);
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
            DISPATCH(BoolConst);
            DISPATCH(Add);
            DISPATCH(Sub);
            DISPATCH(Mul);
            DISPATCH(RealDiv);
            DISPATCH(FloorDiv);
            DISPATCH(CeilDiv);
            DISPATCH(RoundTowards0Div);
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
            DISPATCH(AnyExpr);
        default:
            ERROR("Unexpected Expr node type");
        }
    }
};

} // namespace pybind11
