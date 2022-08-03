#include <analyze/all_uses.h>
#include <pass/make_heap_alloc.h>

namespace freetensor {

Stmt InsertAlloc::visit(const StmtSeq &_op) {
    bool isOuterMostOld = isOuterMost_;
    isOuterMost_ = false;

    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();

    isOuterMost_ = isOuterMostOld;
    if (!isOuterMost_) {
        return op;
    }

    int i = 0;
    while (allUses(op->stmts_[i]).count(var_) == 0) {
        delayed_ = true;
        ++i;
    }

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
    bool isOuterMostOld = isOuterMost_;
    isOuterMost_ = false;

    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();

    isOuterMost_ = isOuterMostOld;
    if (!isOuterMost_) {
        return op;
    }

    int i = op->stmts_.size() - 1;
    while (allUses(op->stmts_[i]).count(var_) == 0) {
        --i;
        madeEarly_ = true;
    }

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

bool MakeHeapAlloc::inKernel() const { return forDepth_ != 0 || inCublas_; }

bool MakeHeapAlloc::isDynamicSized(const VarDef &op) const {
    for (auto &&dim : op->buffer_->tensor()->shape()) {
        if (!dim->isConst()) {
            return true;
        }
    }
    return false;
}

Stmt MakeHeapAlloc::visit(const VarDef &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    if (op->buffer_->atype() != AccessType::Cache ||
        op->buffer_->tensor()->shape().size() == 0) {
        return op;
    }

    switch (op->buffer_->mtype()) {
    case MemType::CPU:
    case MemType::GPUGlobal: {
        // If we can benefit from inserting Alloc and Free, turn them to
        // heap-allocated

        if (op->buffer_->mtype() == MemType::GPUGlobal && inKernel()) {
            return op;
        }

        InsertAlloc insertAlloc(op->name_);
        InsertFree insertFree(op->name_);
        auto newBody = insertAlloc(insertFree(op->body_));
        if (insertAlloc.delayed() || insertFree.madeEarly() ||
            isDynamicSized(op)) {
            switch (op->buffer_->mtype()) {
            case MemType::CPU:
                op->buffer_->setMtype(MemType::CPUHeap);
                break;
            case MemType::GPUGlobal:
                op->buffer_->setMtype(MemType::GPUGlobalHeap);
                break;
            default:
                ASSERT(false);
            }
            op->body_ = newBody;
            return op;
        }
        break;
    }

    case MemType::CPUHeap:
    case MemType::GPUGlobalHeap: {
        // Insert Alloc and Free nodes for already heap-allocated variables

        if (op->buffer_->mtype() == MemType::GPUGlobalHeap && inKernel()) {
            throw InvalidProgram("Unable to allocate a dynamic-sized "
                                 "gpu global memory inside a kernel");
        }

        op->body_ = InsertAlloc(op->name_)(op->body_);
        op->body_ = InsertFree(op->name_)(op->body_);
        break;
    }

    default:;
    }

    return op;
}

Stmt MakeHeapAlloc::visit(const For &_op) {
    bool delta_depth =
        ((_op->property_->parallel_ != serialScope) &&
         std::holds_alternative<CUDAScope>(_op->property_->parallel_));
    if (delta_depth)
        ++forDepth_;
    auto ret = BaseClass::visit(_op);
    if (delta_depth)
        --forDepth_;
    return ret;
}

Stmt MakeHeapAlloc::visit(const MatMul &_op) {
    inCublas_ = true;
    auto ret = BaseClass::visit(_op);
    inCublas_ = false;
    return ret;
}

Stmt makeHeapAlloc(const Stmt &op) { return MakeHeapAlloc()(op); }

} // namespace freetensor
