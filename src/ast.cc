#include <ast.h>
#include <mutator.h>

namespace ir {

std::string toString(ASTNodeType type) {
    switch (type) {
#define DISPATCH(name)                                                         \
    case ASTNodeType::name:                                                    \
        return #name;

        DISPATCH(Func);
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
        ERROR("Unexpected AST node type");
    }
}

std::atomic<uint64_t> StmtNode::idCnt_ = 0;

std::string StmtNode::newId() { return "#" + std::to_string(idCnt_++); }

void StmtNode::setId(const std::string &id) {
    if (id.empty()) {
        id_ = newId();
    } else {
        id_ = id;
    }
}

const std::string &StmtNode::id() const {
    ASSERT(!id_.empty());
    return id_;
}

bool StmtNode::hasNamedId() const { return id_.empty() || id_[0] != '#'; }

Expr deepCopy(const Expr &op) { return Mutator()(op); }
Stmt deepCopy(const Stmt &op) { return Mutator()(op); }

} // namespace ir
