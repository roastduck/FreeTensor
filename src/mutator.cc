#include <except.h>
#include <mutator.h>

namespace ir {

#define DISPATCH_CASE(name)                                                    \
    case ASTNodeType::name:                                                    \
        return visit(op.as<name##Node>());

Expr Mutator::operator()(const Expr &op) {
    switch (op->nodeType()) {
        DISPATCH_CASE(Var);
        DISPATCH_CASE(Load);
        DISPATCH_CASE(IntConst);
        DISPATCH_CASE(FloatConst);
        DISPATCH_CASE(Add);
        DISPATCH_CASE(Sub);
        DISPATCH_CASE(Mul);
        DISPATCH_CASE(Div);
        DISPATCH_CASE(Mod);
        DISPATCH_CASE(LT);
        DISPATCH_CASE(LE);
        DISPATCH_CASE(GT);
        DISPATCH_CASE(GE);
        DISPATCH_CASE(EQ);
        DISPATCH_CASE(NE);

    default:
        ERROR("Unexpected Expr node type");
    }
}

Stmt Mutator::operator()(const Stmt &op) {
    switch (op->nodeType()) {
        DISPATCH_CASE(StmtSeq);
        DISPATCH_CASE(VarDef);
        DISPATCH_CASE(Store);
        DISPATCH_CASE(For);
        DISPATCH_CASE(If);

    default:
        ERROR("Unexpected Stmt node type");
    }
}

} // namespace ir

