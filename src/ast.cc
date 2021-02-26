#include <ast.h>

namespace ir {

std::string toString(ASTNodeType type) {
    switch (type) {
#define DISPATCH(name)                                                         \
    case ASTNodeType::name:                                                    \
        return #name;

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

} // namespace ir

