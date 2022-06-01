#include <algorithm>

#include <ast.h>
#include <hash.h>
#include <mutator.h>

namespace freetensor {

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
        DISPATCH(Assume);
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

AST ASTNode::parentAST() const {
    for (auto p = parent(); p.isValid(); p = p->parent()) {
        if (p->isAST()) {
            return p.as<ASTNode>();
        }
    }
    return nullptr;
}

Expr ExprNode::parentExpr() const {
    for (auto p = parentAST(); p.isValid(); p = p->parentAST()) {
        if (p->isExpr()) {
            return p.as<ExprNode>();
        }
    }
    return nullptr;
}

Stmt StmtNode::parentStmt() const {
    for (auto p = parentAST(); p.isValid(); p = p->parentAST()) {
        if (p->isStmt()) {
            return p.as<StmtNode>();
        }
    }
    return nullptr;
}

Stmt StmtNode::parentCtrlFlow() const {
    for (auto p = parentStmt(); p.isValid(); p = p->parentStmt()) {
        if (p->nodeType() != ASTNodeType::StmtSeq &&
            p->nodeType() != ASTNodeType::VarDef) {
            return p;
        }
    }
    return nullptr;
}

Ref<StmtNode> StmtNode::parentStmtByFilter(
    const std::function<bool(const Stmt &)> &filter) const {
    for (auto p = parentStmt(); p.isValid(); p = p->parentStmt()) {
        if (filter(p)) {
            return p;
        }
    }
    return nullptr;
}

Stmt StmtNode::prevStmt() const {
    if (auto p = parentStmt();
        p.isValid() && p->nodeType() == ASTNodeType::StmtSeq) {
        auto &&seq = p.as<StmtSeqNode>();
        auto it = std::find(seq->stmts_.begin(), seq->stmts_.end(), self());
        ASSERT(it != seq->stmts_.end());
        if (it > seq->stmts_.begin()) {
            return *(it - 1);
        }
    }
    return nullptr;
}

Stmt StmtNode::nextStmt() const {
    if (auto p = parentStmt();
        p.isValid() && p->nodeType() == ASTNodeType::StmtSeq) {
        auto &&seq = p.as<StmtSeqNode>();
        auto it = std::find(seq->stmts_.begin(), seq->stmts_.end(), self());
        ASSERT(it != seq->stmts_.end());
        if (it < seq->stmts_.end() - 1) {
            return *(it + 1);
        }
    }
    return nullptr;
}

Stmt StmtNode::ancestorById(const ID &lookup) const {
    for (auto p = self().as<StmtNode>(); p.isValid(); p = p->parentStmt()) {
        if (p->id() == lookup) {
            return p;
        }
    }
    return nullptr;
}

bool StmtNode::isAncestorOf(const Stmt &other) const {
    if (auto p = other->parentStmt();
        p.isValid() && p->ancestorById(id()).isValid()) {
        return true;
    }
    return false;
}

bool StmtNode::isBefore(const Stmt &other) const {
    auto l = self().as<StmtNode>();
    auto r = other;
    auto common = lcaStmt(l, r);
    if (common->nodeType() == ASTNodeType::StmtSeq) {
        auto &&seq = common.as<StmtSeqNode>();
        while (l->parentStmt() != seq) {
            l = l->parentStmt();
        }
        while (r->parentStmt() != seq) {
            r = r->parentStmt();
        }
        auto itl = std::find(seq->stmts_.begin(), seq->stmts_.end(), l);
        auto itr = std::find(seq->stmts_.begin(), seq->stmts_.end(), r);
        ASSERT(itl != seq->stmts_.end());
        ASSERT(itr != seq->stmts_.end());
        return itl < itr;
    }
    return false;
}

DataType ExprNode::dtype() {
    if (dtype_ == DataType::Invalid) {
        inferDType();
    }
    return dtype_;
}

void ExprNode::resetDType() {
    if (dtype_ != DataType::Invalid) {
        dtype_ = DataType::Invalid;
        if (auto p = parentExpr(); p.isValid()) {
            p->resetDType();
        }
    }
}

ID::ID(const Stmt &stmt) : ID(stmt->id_) {}

const std::string &ID::strId() const {
    if (expr_.isValid()) {
        ERROR("Only Stmt has strId");
    }
    return stmtId_;
}

std::string toString(const ID &id) {
    if (id.expr_.isValid()) {
        return toString(id.expr_) + " in " + id.stmtId_;
    } else {
        return id.stmtId_;
    }
}

bool operator==(const ID &lhs, const ID &rhs) {
    return lhs.stmtId_ == rhs.stmtId_ && HashComparator()(lhs.expr_, rhs.expr_);
}

bool operator!=(const ID &lhs, const ID &rhs) {
    return lhs.stmtId_ != rhs.stmtId_ ||
           !HashComparator()(lhs.expr_, rhs.expr_);
}

std::atomic<uint64_t> StmtNode::idCnt_ = 0;

std::string StmtNode::newId() { return "#" + std::to_string(idCnt_++); }

void StmtNode::setId(const ID &id) {
    if (!id.isValid()) {
        id_ = newId();
    } else {
        if (id.expr_.isValid()) {
            ERROR("Cannot assign an Expr ID to an Stmt");
        }
        id_ = id.stmtId_;
    }
}

ID StmtNode::id() const { return ID(id_); }

bool StmtNode::hasNamedId() const { return id_.empty() || id_[0] != '#'; }

Expr deepCopy(const Expr &op) { return Mutator()(op); }
Stmt deepCopy(const Stmt &op) { return Mutator()(op); }

AST lcaAST(const AST &lhs, const AST &rhs) {
    auto ret = lca(lhs, rhs);
    while (ret.isValid() && !ret->isAST()) {
        ret = ret->parent();
    }
    return ret.as<ASTNode>();
}

Expr lcaExpr(const Expr &lhs, const Expr &rhs) {
    auto ret = lcaAST(lhs, rhs);
    while (ret.isValid() && !ret->isExpr()) {
        ret = ret->parentAST();
    }
    return ret.as<ExprNode>();
}

Stmt lcaStmt(const Stmt &lhs, const Stmt &rhs) {
    auto ret = lcaAST(lhs, rhs);
    while (ret.isValid() && !ret->isStmt()) {
        ret = ret->parentAST();
    }
    return ret.as<StmtNode>();
}

} // namespace freetensor

namespace std {

size_t hash<freetensor::ID>::operator()(const freetensor::ID &id) const {
    return freetensor::hashCombine(freetensor::Hasher()(id.expr_),
                                   std::hash<std::string>()(id.stmtId_));
}

} // namespace std
