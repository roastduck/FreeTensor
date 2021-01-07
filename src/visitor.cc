#include <except.h>
#include <visitor.h>

namespace ir {

void Visitor::operator()(const AST &op) {
    switch (op->nodeType()) {

#define DISPATCH_CASE(name)                                                    \
    case ASTNodeType::name:                                                    \
        visit(op.as<name##Node>());                                            \
        break;

        DISPATCH_CASE(Any);
        DISPATCH_CASE(StmtSeq);
        DISPATCH_CASE(VarDef);
        DISPATCH_CASE(Var);
        DISPATCH_CASE(Store);
        DISPATCH_CASE(Load);
        DISPATCH_CASE(AddTo);
        DISPATCH_CASE(IntConst);
        DISPATCH_CASE(FloatConst);
        DISPATCH_CASE(Add);
        DISPATCH_CASE(Sub);
        DISPATCH_CASE(Mul);
        DISPATCH_CASE(Div);
        DISPATCH_CASE(Mod);
        DISPATCH_CASE(Min);
        DISPATCH_CASE(Max);
        DISPATCH_CASE(LT);
        DISPATCH_CASE(LE);
        DISPATCH_CASE(GT);
        DISPATCH_CASE(GE);
        DISPATCH_CASE(EQ);
        DISPATCH_CASE(NE);
        DISPATCH_CASE(Not);
        DISPATCH_CASE(For);
        DISPATCH_CASE(If);
        DISPATCH_CASE(Assert);

    default:
        ERROR("Unexpected AST node type");
    }
}

void VisitorWithCursor::operator()(const AST &op) {
    switch (op->nodeType()) {
    case ASTNodeType::Any:
    case ASTNodeType::StmtSeq:
    case ASTNodeType::VarDef:
    case ASTNodeType::Store:
    case ASTNodeType::AddTo:
    case ASTNodeType::For:
    case ASTNodeType::If:
    case ASTNodeType::Assert:
        cursor_.enter(op.as<StmtNode>());
        Visitor::operator()(op);
        cursor_.leave();
        break;
    default:
        Visitor::operator()(op);
    }
}

} // namespace ir

