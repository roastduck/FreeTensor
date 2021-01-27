#include <except.h>
#include <visitor.h>

namespace ir {

void Visitor::operator()(const AST &op) {
    switch (op->nodeType()) {

#define DISPATCH_EXPR_CASE(name)                                               \
    case ASTNodeType::name:                                                    \
        visitExpr(op.as<ExprNode>(),                                           \
                  [this](const Expr &_op) { visit(_op.as<name##Node>()); });   \
        break;

#define DISPATCH_STMT_CASE(name)                                               \
    case ASTNodeType::name:                                                    \
        visitStmt(op.as<StmtNode>(),                                           \
                  [this](const Stmt &_op) { visit(_op.as<name##Node>()); });   \
        break;

        DISPATCH_EXPR_CASE(Var);
        DISPATCH_EXPR_CASE(Load);
        DISPATCH_EXPR_CASE(IntConst);
        DISPATCH_EXPR_CASE(FloatConst);
        DISPATCH_EXPR_CASE(Add);
        DISPATCH_EXPR_CASE(Sub);
        DISPATCH_EXPR_CASE(Mul);
        DISPATCH_EXPR_CASE(Div);
        DISPATCH_EXPR_CASE(Mod);
        DISPATCH_EXPR_CASE(Min);
        DISPATCH_EXPR_CASE(Max);
        DISPATCH_EXPR_CASE(LT);
        DISPATCH_EXPR_CASE(LE);
        DISPATCH_EXPR_CASE(GT);
        DISPATCH_EXPR_CASE(GE);
        DISPATCH_EXPR_CASE(EQ);
        DISPATCH_EXPR_CASE(NE);
        DISPATCH_EXPR_CASE(Not);
        DISPATCH_EXPR_CASE(Intrinsic);

        DISPATCH_STMT_CASE(StmtSeq);
        DISPATCH_STMT_CASE(VarDef);
        DISPATCH_STMT_CASE(Store);
        DISPATCH_STMT_CASE(ReduceTo);
        DISPATCH_STMT_CASE(For);
        DISPATCH_STMT_CASE(If);
        DISPATCH_STMT_CASE(Assert);
        DISPATCH_STMT_CASE(Eval);
        DISPATCH_STMT_CASE(Any);

    default:
        ERROR("Unexpected AST node type");
    }
}

} // namespace ir

