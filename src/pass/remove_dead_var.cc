#include <pass/flatten_stmt_seq.h>
#include <pass/remove_dead_var.h>

namespace freetensor {

Stmt RemoveAllWrites::visit(const Store &op) {
    return var_ == op->var_ ? makeStmtSeq({}) : Mutator::visit(op);
}

Stmt RemoveAllWrites::visit(const ReduceTo &op) {
    return var_ == op->var_ ? makeStmtSeq({}) : Mutator::visit(op);
}

Stmt RemoveDeadVar::visitStmt(const Stmt &s) {
    auto ret = BaseClass::visitStmt(s);
    for (auto &&c : ret->children()) {
        if (auto it = writes_.find(c->id()); it != writes_.end()) {
            for (auto &&var : it->second) {
                if (hasDef(var)) { // If still in scope
                    writes_[ret->id()].emplace(var);
                }
            }
        }
    }
    if (auto it = writes_.find(ret->id()); it != writes_.end()) {
        for (auto jt = it->second.begin(); jt != it->second.end();) {
            auto &&var = *jt;
            if (!inLoopCnt_.count(var) && !writtenToOutput_.count(var) &&
                !readsAfterward_.count(var)) {
                ret = RemoveAllWrites{var}(ret);
                jt = it->second.erase(jt);
            } else {
                jt++;
            }
        }
    }
    return ret;
}

Expr RemoveDeadVar::visit(const Load &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    if (destination_ != op->var_) {
        readsAfterward_.insert(op->var_);
        for (auto source = def(op->var_); source->viewOf_.has_value();) {
            source = def(*source->viewOf_);
            readsAfterward_.insert(source->name_);
        }
    }
    return op;
}

Stmt RemoveDeadVar::visit(const Store &op) {
    destination_ = op->var_;
    auto ret = BaseClass::visit(op);
    destination_.clear();
    writes_[ret->id()].emplace(op->var_);
    return ret;
}

Stmt RemoveDeadVar::visit(const ReduceTo &op) {
    destination_ = op->var_;
    auto ret = BaseClass::visit(op);
    destination_.clear();
    writes_[ret->id()].emplace(op->var_);
    return ret;
}

Stmt RemoveDeadVar::visit(const VarDef &_op) {
    bool writtenToOutput = isOutputting(_op->buffer_->atype());
    for (auto source = _op; source->viewOf_.has_value();) {
        source = def(*source->viewOf_);
        if (isOutputting(source->buffer_->atype())) {
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

    if (!writtenToOutput && !readsAfterward_.count(op->name_)) {
        isFixPoint_ = false;
        // If there is no write to this var at all, remove the entire var
        return RemoveAllWrites(op->name_)(op->body_);
    }
    readsAfterward_.erase(op->name_);

    if (op->buffer_->atype() == AccessType::InputMutable &&
        (!writes_.count(op->body_->id()) ||
         !writes_.at(op->body_->id()).count(op->name_))) {
        op->buffer_->setAtype(AccessType::Input);
    }

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

    for (int i = 0;; i++) {
        // There may be redundant reads in an empty For's range or an empty If's
        // condition. Remove these empty nodes first with flatten_stmt_seq. Do
        // it in every iteration.
        op = flattenStmtSeq(op);

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
