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
        DISPATCH_CASE(For);
        DISPATCH_CASE(If);

    default:
        ERROR("Unexpected AST node type");
    }
}

} // namespace ir

