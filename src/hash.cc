#include <itertools.hpp>

#include <hash.h>

namespace freetensor {

size_t Hasher::compHash(const Tensor &t) {
    size_t h = (-1 * K1 + B1) % P;
    for (auto &&dim : t.shape()) {
        h = ((h + dim->hash()) * K2 + B2) % P;
    }
    h = ((h + std::hash<int>()((int)t.dtype())) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const Buffer &b) {
    size_t h = (-1 * K1 + B1) % P;
    h = ((h + b.tensor()->hash()) * K2 + B2) % P;
    h = ((h + std::hash<int>()((int)b.atype())) * K2 + B2) % P;
    h = ((h + std::hash<int>()((int)b.mtype())) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const ReductionItem &r) {
    size_t h = (-1 * K1 + B1) % P;
    h = ((h + std::hash<int>()((int)r.op_)) * K2 + B2) % P;
    h = ((h + std::hash<std::string>()(r.var_)) * K2 + B2) % P;
    for (auto &&idx : r.begins_) {
        h = ((h + (idx.isValid() ? idx->hash() : 0ull)) * K2 + B2) % P;
    }
    for (auto &&idx : r.ends_) {
        h = ((h + (idx.isValid() ? idx->hash() : 0ull)) * K2 + B2) % P;
    }
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const ForProperty &p) {
    size_t h = (-1 * K1 + B1) % P;
    h = ((h + std::hash<ParallelScope>()(p.parallel_)) * K2 + B2) % P;
    h = ((h + std::hash<bool>()(p.unroll_)) * K2 + B2) % P;
    h = ((h + std::hash<bool>()(p.vectorize_)) * K2 + B2) % P;
    for (auto &&r : p.reductions_) {
        h = ((h + r->hash()) * K2 + B2) % P;
    }
    for (auto &&item : p.noDeps_) {
        h = ((h + std::hash<std::string>()(item)) * K2 + B2) % P;
    }
    h = ((h + std::hash<bool>()(p.preferLibs_)) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const AnyNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const StmtSeqNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    for (auto &&stmt : op.stmts_) {
        h = ((h + stmt->hash()) * K2 + B2) % P;
    }
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const VarDefNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + std::hash<std::string>()(op.name_)) * K2 + B2) % P;
    h = ((h + op.buffer_->hash()) * K2 + B2) % P;
    if (op.ioTensor_.isValid()) {
        h = ((h + op.ioTensor_->hash()) * K2 + B2) % P;
    }
    h = ((h + op.body_->hash()) * K2 + B2) % P;
    h = ((h + std::hash<bool>()((int)op.pinned_)) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const StoreNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + std::hash<std::string>()(op.var_)) * K2 + B2) % P;
    for (auto &&index : op.indices_) {
        h = ((h + index->hash()) * K2 + B2) % P;
    }
    h = ((h + op.expr_->hash()) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const ReduceToNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + std::hash<std::string>()(op.var_)) * K2 + B2) % P;
    for (auto &&index : op.indices_) {
        h = ((h + index->hash()) * K2 + B2) % P;
    }
    h = ((h + std::hash<int>()((int)op.op_)) * K2 + B2) % P;
    h = ((h + op.expr_->hash()) * K2 + B2) % P;
    h = ((h + std::hash<bool>()((int)op.atomic_)) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const ForNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + std::hash<std::string>()(op.iter_)) * K2 + B2) % P;
    h = ((h + op.begin_->hash()) * K2 + B2) % P;
    h = ((h + op.end_->hash()) * K2 + B2) % P;
    h = ((h + op.step_->hash()) * K2 + B2) % P;
    h = ((h + op.len_->hash()) * K2 + B2) % P;
    h = ((h + op.property_->hash()) * K2 + B2) % P;
    h = ((h + op.body_->hash()) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const IfNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + op.cond_->hash()) * K2 + B2) % P;
    h = ((h + op.thenCase_->hash()) * K2 + B2) % P;
    if (op.elseCase_.isValid()) {
        h = ((h + op.elseCase_->hash()) * K2 + B2) % P;
    }
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const AssertNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + op.cond_->hash()) * K2 + B2) % P;
    h = ((h + op.body_->hash()) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const AssumeNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + op.cond_->hash()) * K2 + B2) % P;
    h = ((h + op.body_->hash()) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const EvalNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + op.expr_->hash()) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const MatMulNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + op.equivalent_->hash()) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

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
    h = ((h + std::hash<int>()(int(op.destType_))) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

size_t Hasher::compHash(const IntrinsicNode &op) {
    size_t h = ((size_t)op.nodeType() * K1 + B1) % P;
    h = ((h + std::hash<std::string>()(op.format_)) * K2 + B2) % P;
    for (auto &&item : op.params_) {
        h = ((h + item->hash()) * K2 + B2) % P;
    }
    h = ((h + std::hash<int>()(int(op.retType_))) * K2 + B2) % P;
    h = ((h + std::hash<bool>()(op.hasSideEffect_)) * K2 + B2) % P;
    return (h * K3 + B3) % P;
}

bool HashComparator::compare(const Any &lhs, const Any &rhs) const {
    return true;
}

bool HashComparator::compare(const StmtSeq &lhs, const StmtSeq &rhs) const {
    if (lhs->stmts_.size() != rhs->stmts_.size()) {
        return false;
    }
    for (auto &&[l, r] : iter::zip(lhs->stmts_, rhs->stmts_)) {
        if (!(*this)(l, r)) {
            return false;
        }
    }
    return true;
}

bool HashComparator::compare(const VarDef &lhs, const VarDef &rhs) const {
    if (lhs->name_ != rhs->name_) {
        return false;
    }
    if (!(*this)(lhs->buffer_, rhs->buffer_)) {
        return false;
    }
    if (lhs->ioTensor_.isValid() != rhs->ioTensor_.isValid()) {
        return false;
    }
    if (lhs->ioTensor_.isValid() && !(*this)(lhs->ioTensor_, rhs->ioTensor_)) {
        return false;
    }
    if (!(*this)(lhs->body_, rhs->body_)) {
        return false;
    }
    if (lhs->pinned_ != rhs->pinned_) {
        return false;
    }
    return true;
}

bool HashComparator::compare(const Store &lhs, const Store &rhs) const {
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
    if (!(*this)(lhs->expr_, rhs->expr_)) {
        return false;
    }
    return true;
}

bool HashComparator::compare(const ReduceTo &lhs, const ReduceTo &rhs) const {
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
    if (lhs->op_ != rhs->op_) {
        return false;
    }
    if (!(*this)(lhs->expr_, rhs->expr_)) {
        return false;
    }
    if (lhs->atomic_ != rhs->atomic_) {
        return false;
    }
    return true;
}

bool HashComparator::compare(const For &lhs, const For &rhs) const {
    if (lhs->iter_ != rhs->iter_) {
        return false;
    }
    if (!(*this)(lhs->begin_, rhs->begin_)) {
        return false;
    }
    if (!(*this)(lhs->end_, rhs->end_)) {
        return false;
    }
    if (!(*this)(lhs->step_, rhs->step_)) {
        return false;
    }
    if (!(*this)(lhs->len_, rhs->len_)) {
        return false;
    }
    if (!(*this)(lhs->property_, rhs->property_)) {
        return false;
    }
    if (!(*this)(lhs->body_, rhs->body_)) {
        return false;
    }
    return true;
}

bool HashComparator::compare(const If &lhs, const If &rhs) const {
    if (!(*this)(lhs->cond_, rhs->cond_)) {
        return false;
    }
    if (!(*this)(lhs->thenCase_, rhs->thenCase_)) {
        return false;
    }
    if (lhs->elseCase_.isValid() != rhs->elseCase_.isValid()) {
        return false;
    }
    if (lhs->elseCase_.isValid() && !(*this)(lhs->elseCase_, rhs->elseCase_)) {
        return false;
    }
    return true;
}

bool HashComparator::compare(const Assert &lhs, const Assert &rhs) const {
    if (!(*this)(lhs->cond_, rhs->cond_)) {
        return false;
    }
    if (!(*this)(lhs->body_, rhs->body_)) {
        return false;
    }
    return true;
}

bool HashComparator::compare(const Assume &lhs, const Assume &rhs) const {
    if (!(*this)(lhs->cond_, rhs->cond_)) {
        return false;
    }
    if (!(*this)(lhs->body_, rhs->body_)) {
        return false;
    }
    return true;
}

bool HashComparator::compare(const Eval &lhs, const Eval &rhs) const {
    return (*this)(lhs->expr_, rhs->expr_);
}

bool HashComparator::compare(const MatMul &lhs, const MatMul &rhs) const {
    return (*this)(lhs->equivalent_, rhs->equivalent_);
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
    return lhs->destType_ == rhs->destType_ && (*this)(lhs->expr_, rhs->expr_);
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
    if (lhs->hasSideEffect_ != rhs->hasSideEffect_) {
        return false;
    }
    return true;
}

bool HashComparator::operator()(const Ref<Tensor> &lhs,
                                const Ref<Tensor> &rhs) const {
    if (lhs->shape().size() != rhs->shape().size()) {
        return false;
    }
    for (auto &&[l, r] : iter::zip(lhs->shape(), rhs->shape())) {
        if (!(*this)(l, r)) {
            return false;
        }
    }
    if (lhs->dtype() != rhs->dtype()) {
        return false;
    }
    return true;
}

bool HashComparator::operator()(const Ref<Buffer> &lhs,
                                const Ref<Buffer> &rhs) const {
    if (!(*this)(lhs->tensor(), rhs->tensor())) {
        return false;
    }
    if (lhs->mtype() != rhs->mtype()) {
        return false;
    }
    if (lhs->atype() != rhs->atype()) {
        return false;
    }
    return true;
}

bool HashComparator::operator()(const Ref<ReductionItem> &lhs,
                                const Ref<ReductionItem> &rhs) const {
    if (lhs->op_ != rhs->op_) {
        return false;
    }
    if (lhs->var_ != rhs->var_) {
        return false;
    }
    if (lhs->begins_.size() != rhs->begins_.size()) {
        return false;
    }
    for (auto &&[ll, rr] : iter::zip(lhs->begins_, rhs->begins_)) {
        if (!(*this)(ll, rr)) {
            return false;
        }
    }
    if (lhs->ends_.size() != rhs->ends_.size()) {
        return false;
    }
    for (auto &&[ll, rr] : iter::zip(lhs->ends_, rhs->ends_)) {
        if (!(*this)(ll, rr)) {
            return false;
        }
    }
    return true;
}

bool HashComparator::operator()(const Ref<ForProperty> &lhs,
                                const Ref<ForProperty> &rhs) const {
    if (lhs->parallel_ != rhs->parallel_) {
        return false;
    }
    if (lhs->unroll_ != rhs->unroll_) {
        return false;
    }
    if (lhs->vectorize_ != rhs->vectorize_) {
        return false;
    }
    if (lhs->reductions_.size() != rhs->reductions_.size()) {
        return false;
    }
    for (auto &&[l, r] : iter::zip(lhs->reductions_, rhs->reductions_)) {
        if (!(*this)(l, r)) {
            return false;
        }
    }
    if (lhs->noDeps_.size() != rhs->noDeps_.size()) {
        return false;
    }
    for (auto &&[l, r] : iter::zip(lhs->noDeps_, rhs->noDeps_)) {
        if (l != r) {
            return false;
        }
    }
    if (lhs->preferLibs_ != rhs->preferLibs_) {
        return false;
    }
    return true;
}

bool HashComparator::operator()(const AST &lhs, const AST &rhs) const {
    if (lhs == rhs) { // alias or nullptr
        return true;
    }
    if (lhs.isValid() != rhs.isValid()) {
        return false;
    }

    if (lhs->hash() != rhs->hash()) {
        return false;
    }

    if (lhs->nodeType() != rhs->nodeType()) {
        return false;
    }

    if (lhs->isExpr()) {
        if (lhs.as<ExprNode>()->isBinary()) {
            if (lhs.as<BinaryExprNode>()->isCommutative()) {
                return compare(lhs.as<CommutativeBinaryExprNode>(),
                               rhs.as<CommutativeBinaryExprNode>());
            } else {
                return compare(lhs.as<NonCommutativeBinaryExprNode>(),
                               rhs.as<NonCommutativeBinaryExprNode>());
            }
        }
        if (lhs.as<ExprNode>()->isUnary()) {
            return compare(lhs.as<UnaryExprNode>(), rhs.as<UnaryExprNode>());
        }
    }
    switch (lhs->nodeType()) {

#define DISPATCH(name)                                                         \
    case ASTNodeType::name:                                                    \
        return compare(lhs.as<name##Node>(), rhs.as<name##Node>());

        DISPATCH(
            Any); // HashComparator does not treat Any as a universal matcher
        DISPATCH(StmtSeq);
        DISPATCH(VarDef);
        DISPATCH(Store);
        DISPATCH(ReduceTo);
        DISPATCH(For);
        DISPATCH(If);
        DISPATCH(Assert);
        DISPATCH(Assume);
        DISPATCH(Eval);
        DISPATCH(MatMul);
        DISPATCH(Var);
        DISPATCH(Load);
        DISPATCH(IntConst);
        DISPATCH(FloatConst);
        DISPATCH(BoolConst);
        DISPATCH(IfExpr);
        DISPATCH(Cast);
        DISPATCH(Intrinsic);

    default:
        ERROR("Unexpected Expr node type: " + toString(lhs->nodeType()));
    }
}

} // namespace freetensor
