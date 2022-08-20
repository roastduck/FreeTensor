#include <typeinfo>

#include <except.h>
#include <expr.h>
#include <ffi.h>
#include <frontend/frontend_var.h>
#include <func.h>
#include <serialize/load_ast.h>
#include <serialize/print_ast.h>
#include <stmt.h>

namespace freetensor {

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
        .value("Alloc", ASTNodeType::Alloc)
        .value("Free", ASTNodeType::Free)
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
        .value("Assume", ASTNodeType::Assume)
        .value("Intrinsic", ASTNodeType::Intrinsic)
        .value("Eval", ASTNodeType::Eval);

    py::class_<ASTNode, AST> pyAST(m, "AST");
    py::class_<FuncNode, Func>(m, "Func", pyAST);
    py::class_<StmtNode, Stmt>(m, "Stmt", pyAST);
    py::class_<ExprNode, Expr>(m, "Expr", pyAST);

    py::class_<ID>(m, "ID")
        .def(py::init<ID>())
        .def(py::init<>())
        .def("__str__", [](const ID &id) { return toString(id); })
        .def("__hash__", [](const ID &id) { return std::hash<ID>()(id); })
        .def(
            "__eq__", [](const ID &lhs, const ID &rhs) { return lhs == rhs; },
            py::is_operator());

    m.def("make_id", static_cast<ID (*)()>(&ID::make));
    m.def("make_id", static_cast<ID (*)(int64_t)>(&ID::make));

#ifdef FT_DEBUG_LOG_NODE
    pyAST.def_readonly("debug_creator", &ASTNode::debugCreator_);
#endif

    pyAST
        .def("match",
             [](const Stmt &op, const Stmt &other) { return match(op, other); })
        .def("type", [](const AST &op) { return op->nodeType(); })
        .def("node_type",
             [](const AST &op) {
                 WARNING(
                     "`x.node_type()` is deprecated. Please use `x.type()`");
                 return op->nodeType();
             })
        .def("__str__", [](const AST &op) { return toString(op); })
        .def("__repr__", [](const AST &op) {
            return "<" + toString(op->nodeType()) + ": " + toString(op) + ">";
        });
    m.def("dump_ast", &dumpAST, "ast"_a, "dtype_in_load"_a = false);
    m.def("load_ast", &loadAST);
}

} // namespace freetensor

namespace pybind11 {

template <> struct polymorphic_type_hook<freetensor::ASTNode> {
    static const void *get(const freetensor::ASTNode *src,
                           const std::type_info *&type) {
        if (src == nullptr) {
            return src;
        }
        switch (src->nodeType()) {
#define DISPATCH(name)                                                         \
    case freetensor::ASTNodeType::name:                                        \
        type = &typeid(freetensor::name##Node);                                \
        return static_cast<const freetensor::name##Node *>(src);

            DISPATCH(Func);
            DISPATCH(StmtSeq);
            DISPATCH(VarDef);
            DISPATCH(Store);
            DISPATCH(Alloc);
            DISPATCH(Free);
            DISPATCH(ReduceTo);
            DISPATCH(For);
            DISPATCH(If);
            DISPATCH(Assert);
            DISPATCH(Assume);
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

template <> struct polymorphic_type_hook<freetensor::StmtNode> {
    static const void *get(const freetensor::StmtNode *src,
                           const std::type_info *&type) {
        if (src == nullptr) {
            return src;
        }
        switch (src->nodeType()) {
#define DISPATCH(name)                                                         \
    case freetensor::ASTNodeType::name:                                        \
        type = &typeid(freetensor::name##Node);                                \
        return static_cast<const freetensor::name##Node *>(src);

            DISPATCH(StmtSeq);
            DISPATCH(VarDef);
            DISPATCH(Store);
            DISPATCH(ReduceTo);
            DISPATCH(Alloc);
            DISPATCH(Free);
            DISPATCH(For);
            DISPATCH(If);
            DISPATCH(Assert);
            DISPATCH(Assume);
            DISPATCH(Eval);
            DISPATCH(Any);
        default:
            ERROR("Unexpected Stmt node type");
        }
    }
};

template <> struct polymorphic_type_hook<freetensor::ExprNode> {
    static const void *get(const freetensor::ExprNode *src,
                           const std::type_info *&type) {
        if (src == nullptr) {
            return src;
        }
        switch (src->nodeType()) {
#define DISPATCH(name)                                                         \
    case freetensor::ASTNodeType::name:                                        \
        type = &typeid(freetensor::name##Node);                                \
        return static_cast<const freetensor::name##Node *>(src);

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
