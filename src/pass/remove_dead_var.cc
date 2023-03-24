#include <pass/flatten_stmt_seq.h>
#include <pass/remove_dead_var.h>

namespace freetensor {

Stmt RemoveAllWrites::visit(const Store &op) {
    return var_ == op->var_ ? makeStmtSeq({}) : Mutator::visit(op);
}

Stmt RemoveAllWrites::visit(const ReduceTo &op) {
    return var_ == op->var_ ? makeStmtSeq({}) : Mutator::visit(op);
}

Expr RemoveDeadVar::visit(const Load &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    if (destination_ != op->var_) {
        uses_.insert(op->var_);
        for (auto source = def(op->var_); source->viewOf_.has_value();) {
            source = def(*source->viewOf_);
            uses_.insert(source->name_);
        }
    }
    return op;
}

Stmt RemoveDeadVar::visit(const Store &op) {
    if (!inLoopCnt_.count(op->var_) && !writtenToOutput_.count(op->var_) &&
        !uses_.count(op->var_)) {
        // If we are not in a loop, and there is no reads afterwards, remove
        // this write
        return makeStmtSeq({});
    }
    destination_ = op->var_;
    auto ret = BaseClass::visit(op);
    destination_.clear();
    return ret;
}

Stmt RemoveDeadVar::visit(const ReduceTo &op) {
    if (!inLoopCnt_.count(op->var_) && !writtenToOutput_.count(op->var_) &&
        !uses_.count(op->var_)) {
        // If we are not in a loop, and there is no reads afterwards, remove
        // this write
        return makeStmtSeq({});
    }
    destination_ = op->var_;
    auto ret = BaseClass::visit(op);
    destination_.clear();
    return ret;
}

Stmt RemoveDeadVar::visit(const VarDef &_op) {
    bool writtenToOutput = _op->buffer_->atype() == AccessType::Output ||
                           _op->buffer_->atype() == AccessType::InOut;
    for (auto source = _op; source->viewOf_.has_value();) {
        source = def(*source->viewOf_);
        if (source->buffer_->atype() == AccessType::Output ||
            source->buffer_->atype() == AccessType::InOut) {
            writtenToOutput = true;
        }
    }

    if (writtenToOutput) {
        writtenToOutput_.insert(_op->name_);
    }
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    writtenToOutput_.erase(_op->name_);

    if (!writtenToOutput && !uses_.count(op->name_)) {
        isFixPoint_ = false;
        // If there is no write to this var at all, remove the entire var
        return RemoveAllWrites(op->name_)(op->body_);
    }

    uses_.erase(_op->name_);
    return op;
}

Stmt RemoveDeadVar::visit(const For &op) {
    for (auto &&[name, _] : defs()) {
        inLoopCnt_[name]++;
    }
    auto ret = BaseClass::visit(op);
    for (auto &&[name, _] : defs()) {
        if (--inLoopCnt_.at(name) == 0) {
            inLoopCnt_.erase(name);
        }
    }
    return ret;
}

Stmt RemoveDeadVar::visit(const StmtSeq &op) {
    std::vector<Stmt> stmts(op->stmts_.size(), nullptr);
    for (auto &&[s0, s1] : views::reverse(views::zip(op->stmts_, stmts))) {
        s1 = (*this)(s0);
    }
    return makeStmtSeq(std::move(stmts), op->metadata(), op->id());
}

Stmt removeDeadVar(const Stmt &_op) {
    auto op = _op;

    // There may be redundant reads in an empty For's range or an empty If's
    // condition. Remove these empty nodes first with flatten_stmt_seq
    op = flattenStmtSeq(op);

    for (int i = 0;; i++) {
        RemoveDeadVar mutator;
        op = mutator(op);
        if (mutator.isFixPoint() || i > 100) {
            if (i > 100) {
                WARNING("removeDeadVar iterates over 100 rounds. Maybe there "
                        "is a bug");
            }
            break;
        }
    }
    return op;
}

} // namespace freetensor
