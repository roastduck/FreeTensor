#include <schedule/cache.h>

namespace ir {

Stmt MakeCacheVar::visitStmt(
    const Stmt &op, const std::function<Stmt(const Stmt &)> &visitNode) {
    if (op->id() == stmt_) {
        if (!oldBuffer_.isValid()) {
            throw InvalidSchedule("Variable " + oldVar_ + " not found");
        }
        inStmt_ = true;
        auto ret = Mutator::visitStmt(op, visitNode);
        inStmt_ = false;
        Buffer newBuffer(oldBuffer_->tensor(), AccessType::Cache, mtype_);
        ret = makeVarDef("", newVar_, std::move(newBuffer), std::move(ret));
        newDef_ = ret->id();
        return ret;
    } else {
        return Mutator::visitStmt(op, visitNode);
    }
}

Stmt MakeCacheVar::visit(const VarDef &op) {
    if (op->name_ == oldVar_) {
        if (oldBuffer_.isValid()) {
            throw InvalidProgram(
                "Nested VarDef with the same buffer name is not allowed");
        }
        oldBuffer_ = op->buffer_;
        return Mutator::visit(op);
        oldBuffer_ = nullptr;
    } else {
        return Mutator::visit(op);
    }
}

Expr MakeCacheVar::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    if (inStmt_ && op->var_ == oldVar_) {
        op->var_ = newVar_;
    }
    return op;
}

Stmt MakeCacheVar::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    if (inStmt_ && op->var_ == oldVar_) {
        op->var_ = newVar_;
    }
    return op;
}

Stmt MakeCacheVar::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    if (inStmt_ && op->var_ == oldVar_) {
        op->var_ = newVar_;
    }
    return op;
}

Stmt MakeFillAndFlush::visitStmt(
    const Stmt &_op, const std::function<Stmt(const Stmt &)> &visitNode) {
    auto op = Mutator::visitStmt(_op, visitNode);
    if (op->id() == stmt_) {
        std::vector<std::string> iters;
        std::vector<Expr> indices;
        ASSERT(nDim_ != -1);
        iters.reserve(nDim_);
        indices.reserve(nDim_);
        for (int i = 0; i < nDim_; i++) {
            std::string iter = "." + newVar_ + ".i" + std::to_string(i);
            indices.emplace_back(makeVar(iter));
            iters.emplace_back(std::move(iter));
        }

        Stmt fill;
        if (rRange_.count(newDef_)) {
            auto &&rRange = rRange_.at(newDef_);
            fill = makeStore("", newVar_, indices, makeLoad(oldVar_, indices));
            for (int i = nDim_ - 1; i >= 0; i--) {
                fill = makeFor("", iters[i], rRange.lower_[i],
                               makeAdd(rRange.lower_[i], rRange.len_[i]), "",
                               fill);
            }
        } else {
            fill = makeStmtSeq("", {});
        }

        Stmt flush;
        if (wRange_.count(newDef_)) {
            auto &&wRange = wRange_.at(newDef_);
            flush = makeStore("", oldVar_, indices, makeLoad(newVar_, indices));
            for (int i = nDim_ - 1; i >= 0; i--) {
                flush = makeFor("", iters[i], wRange.lower_[i],
                                makeAdd(wRange.lower_[i], wRange.len_[i]), "",
                                flush);
            }
        } else {
            flush = makeStmtSeq("", {});
        }

        fillStmt_ = fill->id();
        flushStmt_ = flush->id();
        op = makeStmtSeq("", {fill, op, flush});
    }
    return op;
}

Stmt MakeFillAndFlush::visit(const VarDef &op) {
    if (op->id() == newDef_) {
        nDim_ = op->buffer_->tensor().shape().size();
        auto ret = Mutator::visit(op);
        nDim_ = -1;
        return ret;
    } else {
        return Mutator::visit(op);
    }
}

} // namespace ir

