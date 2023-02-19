#include <algorithm>

#include <ast.h>
#include <hash.h>
#include <mutator.h>

namespace freetensor {

std::ostream &operator<<(std::ostream &os, ASTNodeType type) {
    switch (type) {
#define DISPATCH(name)                                                         \
    case ASTNodeType::name:                                                    \
        return os << #name;

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
        DISPATCH(MarkVersion);
        DISPATCH(LoadAtVersion);
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

Stmt ExprNode::parentStmt() const {
    for (auto p = parentAST(); p.isValid(); p = p->parentAST()) {
        if (p->isStmt()) {
            return p.as<StmtNode>();
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
    if (auto p = parentStmt(); p.isValid()) {
        auto &&children = p->children();
        auto it = std::find(children.begin(), children.end(), self());
        ASSERT(it != children.end());
        if (it > children.begin()) {
            return *(it - 1);
        }
    }
    return nullptr;
}

Stmt StmtNode::nextStmt() const {
    if (auto p = parentStmt(); p.isValid()) {
        auto &&children = p->children();
        auto it = std::find(children.begin(), children.end(), self());
        ASSERT(it != children.end());
        if (it < children.end() - 1) {
            return *(it + 1);
        }
    }
    return nullptr;
}

Stmt StmtNode::parentCtrlFlow() const {
    for (auto p = parentStmt(); p.isValid(); p = p->parentStmt()) {
        if (p->isCtrlFlow()) {
            return p;
        }
    }
    return nullptr;
}

Stmt StmtNode::prevInCtrlFlow() const {
    Stmt ret = self().as<StmtNode>();
    while (!ret->prevStmt().isValid()) {
        if (auto p = ret->parentStmt(); p.isValid() && !p->isCtrlFlow()) {
            ret = p;
        } else {
            return nullptr;
        }
    }
    ret = ret->prevStmt();
    while (!ret->isCtrlFlow()) {
        if (auto &&children = ret->children(); !children.empty()) {
            ret = children.back();
        } else {
            break;
        }
    }
    return ret;
}

Stmt StmtNode::nextInCtrlFlow() const {
    Stmt ret = self().as<StmtNode>();
    while (!ret->nextStmt().isValid()) {
        if (auto p = ret->parentStmt(); p.isValid() && !p->isCtrlFlow()) {
            ret = p;
        } else {
            return nullptr;
        }
    }
    ret = ret->nextStmt();
    while (!ret->isCtrlFlow()) {
        if (auto &&children = ret->children(); !children.empty()) {
            ret = children.front();
        } else {
            break;
        }
    }
    return ret;
}

Stmt StmtNode::prevStmtInDFSOrder() const {
    auto node = self().as<StmtNode>();
    while (true) {
        if (auto &&p = node->prevStmt(); p.isValid()) {
            node = p;
            break;
        }
        if (auto &&p = node->parentStmt(); p.isValid()) {
            node = p;
        } else {
            return nullptr;
        }
    }
    while (true) {
        if (auto &&ch = node->children(); !ch.empty()) {
            node = ch.back();
        } else {
            break;
        }
    }
    return node;
}

Stmt StmtNode::nextStmtInDFSOrder() const {
    auto node = self().as<StmtNode>();
    while (true) {
        if (auto &&p = node->nextStmt(); p.isValid()) {
            node = p;
            break;
        }
        if (auto &&p = node->parentStmt(); p.isValid()) {
            node = p;
        } else {
            return nullptr;
        }
    }
    while (true) {
        if (auto &&ch = node->children(); !ch.empty()) {
            node = ch.front();
        } else {
            break;
        }
    }
    return node;
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
    if (l == common || r == common) {
        // There is no before or after relation between a node and its children
        return false;
    }
    while (l->parentStmt() != common) {
        l = l->parentStmt();
    }
    while (r->parentStmt() != common) {
        r = r->parentStmt();
    }
    if (common->nodeType() == ASTNodeType::StmtSeq) {
        auto &&seq = common.as<StmtSeqNode>();
        auto itl = std::find(seq->stmts_.begin(), seq->stmts_.end(), l);
        auto itr = std::find(seq->stmts_.begin(), seq->stmts_.end(), r);
        ASSERT(itl != seq->stmts_.end());
        ASSERT(itr != seq->stmts_.end());
        return itl < itr;
    } else if (common->nodeType() == ASTNodeType::If) {
        // We treat the then-case is BEFORE the else-case. Otherwise, user would
        // have problem specifying the fission point (before or after a
        // statement) when applying a `fission` schedule.
        auto &&branch = common.as<IfNode>();
        if (l == branch->thenCase_ && r == branch->elseCase_) {
            return true;
        }
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

std::ostream &operator<<(std::ostream &os, const StmtOrExprID &id) {
    if (id.expr_.isValid())
        return os << id.expr_ << " in " << id.stmtId_;
    else
        return os << id.stmtId_;
}

bool operator==(const StmtOrExprID &lhs, const StmtOrExprID &rhs) {
    return lhs.stmtId_ == rhs.stmtId_ && HashComparator()(lhs.expr_, rhs.expr_);
}

void StmtNode::setId(const ID &id) {
    if (!id.isValid())
        id_ = ID::make();
    else
        id_ = id;
}
ID StmtNode::id() const { return id_; }

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

size_t hash<freetensor::StmtOrExprID>::operator()(
    const freetensor::StmtOrExprID &id) const {
    return freetensor::hashCombine(freetensor::Hasher()(id.expr_),
                                   std::hash<freetensor::ID>()(id.stmtId_));
}

} // namespace std
