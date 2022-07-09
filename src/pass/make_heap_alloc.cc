#include <analyze/all_uses.h>
#include <pass/make_heap_alloc.h>

namespace freetensor {

Stmt InsertAlloc::visit(const StmtSeq &_op) {
    bool tmp_insert = is_insert;
    is_insert = false;

    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();

    if (!tmp_insert)
        return op;
    is_insert = true;

    int i = 0;
    while (allUses(op->stmts_[i]).count(var_) == 0)
        ++i;

    ASSERT(i < (int)op->stmts_.size());

    if (op->stmts_[i]->nodeType() == ASTNodeType::For ||
        op->stmts_[i]->nodeType() == ASTNodeType::If ||
        op->stmts_[i]->nodeType() == ASTNodeType::Store ||
        op->stmts_[i]->nodeType() == ASTNodeType::Load ||
        op->stmts_[i]->nodeType() == ASTNodeType::ReduceTo ||
        op->stmts_[i]->isExpr()) {
        op->stmts_.insert(op->stmts_.begin() + i, makeAlloc(op->id(), var_));
    } else {
        op->stmts_[i] = (*this)(op->stmts_[i]);
    }

    return op;
}

Stmt InsertFree::visit(const StmtSeq &_op) {
    bool tmp_insert = is_insert;
    is_insert = false;

    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();

    if (!tmp_insert)
        return op;
    is_insert = true;

    int i = op->stmts_.size() - 1;
    while (allUses(op->stmts_[i]).count(var_) == 0)
        --i;

    ASSERT(i >= 0);

    if (op->stmts_[i]->nodeType() == ASTNodeType::For ||
        op->stmts_[i]->nodeType() == ASTNodeType::If ||
        op->stmts_[i]->nodeType() == ASTNodeType::Store ||
        op->stmts_[i]->nodeType() == ASTNodeType::Load ||
        op->stmts_[i]->nodeType() == ASTNodeType::ReduceTo ||
        op->stmts_[i]->isExpr()) {
        op->stmts_.insert(op->stmts_.begin() + i + 1, makeFree(op->id(), var_));
    } else {
        op->stmts_[i] = (*this)(op->stmts_[i]);
    }

    return op;
}

bool MakeHeapAlloc::inKernel() const { return for_depth != 0 || inCublas_; }

Stmt MakeHeapAlloc::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    if (op->buffer_->mtype() == MemType::GPUGlobalHeap) {
        ASSERT(!inKernel());
    }

    if (op->buffer_->mtype() == MemType::CPUHeap ||
        op->buffer_->mtype() == MemType::GPUGlobalHeap) {
        if (op->buffer_->tensor()->shape().size() != 0) {
            op->body_ = InsertAlloc(op->name_)(op->body_);
            op->body_ = InsertFree(op->name_)(op->body_);
        }
    }

    return op;
}

Stmt MakeHeapAlloc::visit(const For &_op) {
    bool delta_depth =
        ((_op->property_->parallel_ != serialScope) &&
         std::holds_alternative<CUDAScope>(_op->property_->parallel_));
    if (delta_depth)
        ++for_depth;
    auto __op = Mutator::visit(_op);
    if (delta_depth)
        --for_depth;
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    return op;
}

Stmt MakeHeapAlloc::visit(const MatMul &_op) {
    inCublas_ = true;
    auto __op = Mutator::visit(_op);
    inCublas_ = false;
    ASSERT(__op->nodeType() == ASTNodeType::MatMul);
    auto op = __op.as<MatMulNode>();
    return op;
}

Stmt makeHeapAlloc(const Stmt &_op) {
    auto op = _op;
    MakeHeapAlloc mutator;
    op = mutator(op);
    return op;
}

} // namespace freetensor
