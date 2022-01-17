#include <itertools.hpp>

#include <hash.h>

namespace ir {

size_t Hasher::compHash(const CommutativeBinaryExprNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h += op.lhs_->hash() + op.rhs_->hash();
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const NonCommutativeBinaryExprNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + op.lhs_->hash()) * K2 + B2) % P;
    h = ((h + op.rhs_->hash()) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const UnaryExprNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + op.expr_->hash()) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const AnyExprNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const VarNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + std::hash<std::string>()(op.name_)) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const LoadNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + std::hash<std::string>()(op.var_)) * K2 + B2) % P;
    for (auto &&index : op.indices_) {
        h = ((h + index->hash()) * K2 + B2) % P;
    }
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const IntConstNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + std::hash<int>()(op.val_)) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const FloatConstNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + std::hash<double>()(op.val_)) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const BoolConstNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + std::hash<bool>()(op.val_)) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const IfExprNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + op.cond_->hash()) * K2 + B2) % P;
    h = ((h + op.thenCase_->hash()) * K2 + B2) % P;
    h = ((h + op.elseCase_->hash()) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const CastNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + op.expr_->hash()) * K2 + B2) % P;
    h = ((h + std::hash<int>()(int(op.dtype_))) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const IntrinsicNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + std::hash<std::string>()(op.format_)) * K2 + B2) % P;
    for (auto &&item : op.params_) {
        h = ((h + item->hash()) * K2 + B2) % P;
    }
    return (h * K3 + B3) % P;
}

bool HashComparator::compare(const CommutativeBinaryExpr &lhs,
                             const CommutativeBinaryExpr &rhs) const {
    if ((lhs->lhs_->hash() < lhs->rhs_->hash()) ==
        (rhs->lhs_->hash() < rhs->rhs_->hash())) {
        return (*this)(lhs->lhs_, rhs->lhs_) && (*this)(lhs->rhs_, rhs->rhs_);
    } else {
        return (*this)(lhs->lhs_, rhs->rhs_) && (*this)(lhs->rhs_, rhs->lhs_);
    }
}

bool HashComparator::compare(const NonCommutativeBinaryExpr &lhs,
                             const NonCommutativeBinaryExpr &rhs) const {
    return (*this)(lhs->lhs_, rhs->lhs_) && (*this)(lhs->rhs_, rhs->rhs_);
}

bool HashComparator::compare(const UnaryExpr &lhs, const UnaryExpr &rhs) const {
    return (*this)(lhs->expr_, rhs->expr_);
}

bool HashComparator::compare(const Var &lhs, const Var &rhs) const {
    return lhs->name_ == rhs->name_;
}

bool HashComparator::compare(const IntConst &lhs, const IntConst &rhs) const {
    return lhs->val_ == rhs->val_;
}

bool HashComparator::compare(const FloatConst &lhs,
                             const FloatConst &rhs) const {
    return lhs->val_ == rhs->val_;
}

bool HashComparator::compare(const BoolConst &lhs, const BoolConst &rhs) const {
    return lhs->val_ == rhs->val_;
}

bool HashComparator::compare(const Load &lhs, const Load &rhs) const {
    if (lhs->var_ != rhs->var_) {
        return false;
    }
    if (lhs->indices_.size() != rhs->indices_.size()) {
        return false;
    }
    for (auto &&[l, r] : iter::zip(lhs->indices_, rhs->indices_)) {
        if (!(*this)(l, r)) {
            return false;
        }
    }
    return true;
}

bool HashComparator::compare(const IfExpr &lhs, const IfExpr &rhs) const {
    return (*this)(lhs->cond_, rhs->cond_) &&
           (*this)(lhs->thenCase_, rhs->thenCase_) &&
           (*this)(lhs->elseCase_, rhs->elseCase_);
}

bool HashComparator::compare(const Cast &lhs, const Cast &rhs) const {
    return lhs->dtype_ == rhs->dtype_ && (*this)(lhs->expr_, rhs->expr_);
}

bool HashComparator::compare(const Intrinsic &lhs, const Intrinsic &rhs) const {
    if (lhs->format_ != rhs->format_) {
        return false;
    }
    if (lhs->params_.size() != rhs->params_.size()) {
        return false;
    }
    for (auto &&[l, r] : iter::zip(lhs->params_, rhs->params_)) {
        if (!(*this)(l, r)) {
            return false;
        }
    }
    if (lhs->retType_ != rhs->retType_) {
        return false;
    }
    return true;
}

bool HashComparator::operator()(const Expr &lhs, const Expr &rhs) const {
    if (lhs->hash() != rhs->hash()) {
        return false;
    }

    if (lhs->nodeType() != rhs->nodeType()) {
        return false;
    }

    if (lhs->isBinary()) {
        if (lhs.as<BinaryExprNode>()->isCommutative()) {
            return compare(lhs.as<CommutativeBinaryExprNode>(),
                           rhs.as<CommutativeBinaryExprNode>());
        } else {
            return compare(lhs.as<NonCommutativeBinaryExprNode>(),
                           rhs.as<NonCommutativeBinaryExprNode>());
        }
    }
    if (lhs->isUnary()) {
        return compare(lhs.as<UnaryExprNode>(), rhs.as<UnaryExprNode>());
    }
    switch (lhs->nodeType()) {

#define DISPATCH(name)                                                         \
    case ASTNodeType::name:                                                    \
        return compare(lhs.as<name##Node>(), rhs.as<name##Node>());

        DISPATCH(Var);
        DISPATCH(Load);
        DISPATCH(IntConst);
        DISPATCH(FloatConst);
        DISPATCH(BoolConst);
        DISPATCH(IfExpr);
        DISPATCH(Cast);
        DISPATCH(Intrinsic);

    default:
        ERROR("Unexpected Expr node type");
    }
}

} // namespace ir

