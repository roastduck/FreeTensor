#include <typeinfo>

#include <debug.h>
#include <except.h>
#include <expr.h>
#include <ffi.h>
#include <func.h>
#include <stmt.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_ast(py::module_ &m) {
    py::enum_<ASTNodeType>(m, "ASTNodeType")
        .value("Any", ASTNodeType::Any)
        .value("AnyExpr", ASTNodeType::AnyExpr)
        .value("Func", ASTNodeType::Func)
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
        .value("Remainder", ASTNodeType::Remainder)
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
        .value("Sqrt", ASTNodeType::Sqrt)
        .value("Exp", ASTNodeType::Exp)
        .value("Square", ASTNodeType::Square)
        .value("Sigmoid", ASTNodeType::Sigmoid)
        .value("Tanh", ASTNodeType::Tanh)
        .value("Abs", ASTNodeType::Abs)
        .value("Floor", ASTNodeType::Floor)
        .value("Ceil", ASTNodeType::Ceil)
        .value("IfExpr", ASTNodeType::IfExpr)
        .value("Cast", ASTNodeType::Cast)
        .value("For", ASTNodeType::For)
        .value("If", ASTNodeType::If)
        .value("Assert", ASTNodeType::Assert)
        .value("Intrinsic", ASTNodeType::Intrinsic)
        .value("Eval", ASTNodeType::Eval);

    py::class_<ASTNode, AST> pyAST(m, "AST");
    py::class_<FuncNode, Func> pyFunc(m, "Func", pyAST);
    py::class_<StmtNode, Stmt> pyStmt(m, "Stmt", pyAST);
    py::class_<ExprNode, Expr> pyExpr(m, "Expr", pyAST);

#ifdef IR_DEBUG_LOG_NODE
    pyAST.def_readonly("debug_creator", &ASTNode::debugCreator_);
#endif

    pyFunc.def_readonly("name", &FuncNode::name_)
        .def_readonly("params", &FuncNode::params_)
        .def_readonly("returns", &FuncNode::returns_)
        .def_property_readonly(
            "body", [](const Func &op) -> Stmt { return op->body_; });

    py::class_<FrontendVarIdx>(m, "FrontendVarIdx")
        .def(py::init(&FrontendVarIdx::fromSingle))
        .def(py::init(&FrontendVarIdx::fromSlice))
        .def("__repr__",
             [](const FrontendVarIdx &idx) { return toString(idx); });

    py::class_<FrontendVar, Ref<FrontendVar>>(m, "FrontendVar")
        .def(py::init<const std::string &, const std::vector<Expr> &, DataType,
                      MemType, const std::vector<FrontendVarIdx> &>())
        .def_property_readonly("name", &FrontendVar::name)
        .def_property_readonly("full_shape", &FrontendVar::fullShape)
        .def_property_readonly("indices", &FrontendVar::indices)
        .def_property_readonly("dtype", &FrontendVar::dtype)
        .def_property_readonly("mtype", &FrontendVar::mtype)
        .def_property_readonly("ndim", &FrontendVar::ndim)
        .def("shape", &FrontendVar::shape)
        .def("as_load", &FrontendVar::asLoad)
        .def("as_store", &FrontendVar::asStore)
        .def("chain_indices", &FrontendVar::chainIndices)
        .def("__repr__", [](const FrontendVar &var) { return toString(var); });

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
    py::class_<ForProperty>(m, "ForProperty")
        .def(py::init<>())
        .def_readonly("parallel", &ForProperty::parallel_)
        .def_readonly("unroll", &ForProperty::unroll_)
        .def_readonly("vectorize", &ForProperty::vectorize_)
        .def_readonly("no_deps", &ForProperty::noDeps_)
        .def_readonly("prefer_libs", &ForProperty::preferLibs_)
        .def("with_parallel", &ForProperty::withParallel, "parallel"_a)
        .def("with_unroll", &ForProperty::withUnroll, "unroll"_a = true)
        .def("with_vectorize", &ForProperty::withVectorize,
             "vectorize"_a = true)
        .def("with_no_deps", &ForProperty::withNoDeps, "var"_a)
        .def("with_prefer_libs", &ForProperty::withPreferLibs, "prefer_libs"_a);
    py::class_<ForNode, For>(m, "For", pyStmt)
        .def_readonly("iter", &ForNode::iter_)
        .def_property_readonly("begin",
                               [](const For &op) -> Expr { return op->begin_; })
        .def_property_readonly("end",
                               [](const For &op) -> Expr { return op->end_; })
        .def_property_readonly("step",
                               [](const For &op) -> Expr { return op->step_; })
        .def_property_readonly("len",
                               [](const For &op) -> Expr { return op->len_; })
        .def_readonly("property", &ForNode::property_)
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

    py::class_<VarNode, Var>(m, "Var", pyExpr)
        .def_readonly("name", &VarNode::name_);
    py::class_<LoadNode, Load>(m, "Load", pyExpr)
        .def_readonly("var", &LoadNode::var_)
        .def_property_readonly(
            "indices", [](const Load &op) -> std::vector<Expr> {
                return std::vector<Expr>(op->indices_.begin(),
                                         op->indices_.end());
            });
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
        .def_property_readonly(
            "dtype", [](const Cast &op) -> DataType { return op->dtype_; });
    py::class_<IntrinsicNode, Intrinsic> pyIntrinsic(m, "Intrinsic", pyExpr);
    py::class_<AnyExprNode, AnyExpr> pyAnyExpr(m, "AnyExpr", pyExpr);

    pyAST
        .def("match",
             [](const AST &op, const AST &other) { return match(op, other); })
        .def("type", [](const AST &op) { return toString(op->nodeType()); })
        .def("__str__", [](const AST &op) { return toString(op); })
        .def("__repr__",
             [](const AST &op) {
                 return "<" + toString(op->nodeType()) + ": " + toString(op) +
                        ">";
             })
        .def("pretty_print", [](const AST &op) { return toString(op, true); });

    // NOTE: ORDER of the constructor matters!
    pyExpr.def(py::init([](bool val) { return makeBoolConst(val); }))
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
            py::is_operator());
    py::implicitly_convertible<int, ExprNode>();
    py::implicitly_convertible<float, ExprNode>();
    py::implicitly_convertible<bool, ExprNode>();
    py::implicitly_convertible<FrontendVar, ExprNode>();

    // Function
    m.def(
        "makeFunc",
        [](const std::string &name, const std::vector<std::string> &params,
           const std::vector<std::pair<std::string, DataType>> &returns,
           const Stmt &body,
           const std::unordered_map<std::string, Ref<Array>> &_closure) {
            std::unordered_map<std::string, Ref<Ref<Array>>> closure;
            for (auto &&[name, var] : _closure) {
                closure[name] = Ref<Ref<Array>>::make(var);
            }
            return makeFunc(name, params, returns, body, closure);
        },
        "name"_a, "params"_a, "returns"_a, "body"_a, "closure"_a);

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
    m.def("makeLoad",
          static_cast<Expr (*)(const std::string &, const std::vector<Expr> &)>(
              &_makeLoad),
          "var"_a, "indices"_a);
    m.def("makeIntConst", &_makeIntConst, "val"_a);
    m.def("makeFloatConst", &_makeFloatConst, "val"_a);
    m.def(
        "makeFor",
        static_cast<Stmt (*)(const std::string &, const std::string &,
                             const Expr &, const Expr &, const Expr &,
                             const Expr &, const ForProperty &, const Stmt &)>(
            &_makeFor),
        "nid"_a, "iter"_a, "begin"_a, "end"_a, "step"_a, "len"_a, "property"_a,
        "body"_a);
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

    m.def("neutral_val", &neutralVal);
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
            DISPATCH(Remainder);
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
            DISPATCH(Sqrt);
            DISPATCH(Exp);
            DISPATCH(Square);
            DISPATCH(Sigmoid);
            DISPATCH(Tanh);
            DISPATCH(Abs);
            DISPATCH(Floor);
            DISPATCH(Ceil);
            DISPATCH(IfExpr);
            DISPATCH(Cast);
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
            DISPATCH(Remainder);
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
            DISPATCH(Sqrt);
            DISPATCH(Exp);
            DISPATCH(Square);
            DISPATCH(Sigmoid);
            DISPATCH(Tanh);
            DISPATCH(Abs);
            DISPATCH(Floor);
            DISPATCH(Ceil);
            DISPATCH(IfExpr);
            DISPATCH(Cast);
            DISPATCH(Intrinsic);
            DISPATCH(AnyExpr);
        default:
            ERROR("Unexpected Expr node type");
        }
    }
};

} // namespace pybind11
