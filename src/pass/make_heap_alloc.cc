#include <pass/make_heap_alloc.h>
#include <analyze/all_uses.h>

namespace freetensor {

Stmt InsertAlloc::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<StmtSeqNode>();
    
    int i = 0;
    while (allUses(op->stmts_[i]).count(var_) == 0) ++ i;

    assert(i < (int)op->stmts_.size());
    
    if (op->stmts_[i]->nodeType() == ASTNodeType::For ||
        op->stmts_[i]->nodeType() == ASTNodeType::If ||
        op->stmts_[i]->isExpr()) {
        op->stmts_.insert(op->stmts_.begin() + i, makeAlloc(op->id(), var_));
    } else {
        (*this)(op->stmts_[i]);
    }
    
    return op;
}

Stmt InsertFree::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<StmtSeqNode>();

    int i = op->stmts_.size() - 1;
    while (allUses(op->stmts_[i]).count(var_) == 0) -- i;

    assert(i >= 0);
    
    if (op->stmts_[i]->nodeType() == ASTNodeType::For ||
        op->stmts_[i]->nodeType() == ASTNodeType::If ||
        op->stmts_[i]->isExpr()) {
        op->stmts_.insert(op->stmts_.begin() + i, makeAlloc(op->id(), var_));
    } else {
        (*this)(op->stmts_[i]);
    }

    return op;
}

Stmt InsertAllocFree::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<StmtSeqNode>();

    int i = 0, j = op->stmts_.size() - 1;
    while (i < (int)op->stmts_.size() && allUses(op->stmts_[i]).count(var_) == 0) ++ i;
    while (j >= 0 && allUses(op->stmts_[j]).count(var_) == 0) -- j;

    if (i == j) {
        (*this)(op->stmts_[i]);
    } else {
        InsertAlloc(this->var_)(op->stmts_[i]);
        InsertFree(this->var_)(op->stmts_[i]);
    }

    return op;
}

Stmt MakeHeapAlloc::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    if (op->buffer_->mtype() == MemType::CPUHeap or
        op->buffer_->mtype() == MemType::GPUGlobalHeap) {
        return InsertAllocFree(op->name_)(op->body_);
    }

    return op;
}

Stmt makeHeapAlloc(const Stmt &_op) {
    auto op = _op;
    MakeHeapAlloc mutator;
    op = mutator(op);
    return op;
}

} // namespace freetensor
