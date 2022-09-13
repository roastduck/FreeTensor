#include <analyze/all_uses.h>
#include <pass/make_heap_alloc.h>

namespace freetensor {

Stmt InsertAlloc::visit(const StmtSeq &op) {
    for (size_t i = 0, n = op->stmts_.size(); i < n; i++) {
        if (allUses(op->stmts_[i]).count(var_)) {
            std::vector<Stmt> stmts = op->stmts_;
            stmts[i] = (*this)(op->stmts_[i]);
            if (!inserted_) {
                stmts.insert(stmts.begin() + i, makeAlloc(var_));
                inserted_ = true;
            }
            return makeStmtSeq(std::move(stmts));
        } else {
            delayed_ = true;
        }
    }
    ERROR("Variable defined but not used");
}

Stmt InsertFree::visit(const StmtSeq &op) {
    for (size_t i = op->stmts_.size() - 1; ~i; i--) {
        if (allUses(op->stmts_[i]).count(var_)) {
            std::vector<Stmt> stmts = op->stmts_;
            stmts[i] = (*this)(op->stmts_[i]);
            if (!inserted_) {
                stmts.insert(stmts.begin() + i + 1, makeFree(var_));
                inserted_ = true;
            }
            return makeStmtSeq(std::move(stmts));
        } else {
            madeEarly_ = true;
        }
    }
    ERROR("Variable defined but not used");
}

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

    if (!allUses(op->body_).count(op->name_)) {
        return op->body_;
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
    bool oldInKernel = inKernel_;
    if (std::holds_alternative<CUDAScope>(_op->property_->parallel_)) {
        inKernel_ = true;
    }
    auto ret = BaseClass::visit(_op);
    inKernel_ = oldInKernel;
    return ret;
}

Stmt makeHeapAlloc(const Stmt &op) { return MakeHeapAlloc()(op); }

} // namespace freetensor
