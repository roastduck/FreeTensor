#include <climits>
#include <cmath>

#include <schedule/cache.h>

namespace ir {

static Expr makeNeutralVal(DataType dtype, ReduceOp op) {
    switch (dtype) {
    case DataType::Float32:
        switch (op) {
        case ReduceOp::Add:
            return makeFloatConst(0.);
        case ReduceOp::Max:
            return makeFloatConst(-INFINITY);
        case ReduceOp::Min:
            return makeFloatConst(INFINITY);
        default:
            ASSERT(false);
        }

    case DataType::Int32:
        switch (op) {
        case ReduceOp::Add:
            return makeIntConst(0);
        case ReduceOp::Max:
            return makeIntConst(INT_MIN);
        case ReduceOp::Min:
            return makeIntConst(INT_MAX);
        default:
            ASSERT(false);
        }

    default:
        ASSERT(false);
    }
}

Stmt MakeCacheVar::visitStmt(
    const Stmt &op, const std::function<Stmt(const Stmt &)> &visitNode) {
    if (op->id() == stmt_) {
        if (!def_.isValid()) {
            throw InvalidSchedule("Variable " + oldVar_ + " not found");
        }
        inStmt_ = true;
        auto ret = Mutator::visitStmt(op, visitNode);
        inStmt_ = false;
        Buffer newBuffer(def_->buffer_->tensor(), AccessType::Cache, mtype_);
        ret = makeVarDef("", newVar_, std::move(newBuffer), nullptr,
                         std::move(ret));
        oldDef_ = def_->id();
        newDef_ = ret->id();
        return ret;
    } else {
        return Mutator::visitStmt(op, visitNode);
    }
}

Stmt MakeCacheVar::visit(const VarDef &op) {
    if (op->name_ == oldVar_) {
        if (def_.isValid()) {
            throw InvalidProgram(
                "Nested VarDef with the same buffer name is not allowed");
        }
        def_ = op;
        return Mutator::visit(op);
        def_ = nullptr;
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
        ASSERT(def_.isValid());
        int nDim = def_->buffer_->tensor().shape().size();
        iters.reserve(nDim);
        indices.reserve(nDim);
        for (int i = 0; i < nDim; i++) {
            std::string iter = "." + newVar_ + ".i" + std::to_string(i);
            indices.emplace_back(makeVar(iter));
            iters.emplace_back(std::move(iter));
        }

        Expr idx1d;
        if (def_->sizeLim_.isValid()) {
            auto &&shape = def_->buffer_->tensor().shape();
            for (int i = 0; i < nDim; i++) {
                idx1d = idx1d.isValid() ? makeMul(idx1d, shape[i]) : nullptr;
                idx1d =
                    idx1d.isValid() ? makeAdd(idx1d, indices[i]) : indices[i];
            }
        }

        Stmt fill;
        if (rRange_.count(newDef_)) {
            auto &&rRange = rRange_.at(newDef_);
            fill = makeStore("", newVar_, indices, makeLoad(oldVar_, indices));
            fillStmt_ = fill->id();
            if (idx1d.isValid()) {
                fill = makeIf("", makeLT(idx1d, def_->sizeLim_), fill);
            }
            for (int i = nDim - 1; i >= 0; i--) {
                fill = makeFor("", iters[i], rRange.lower_[i],
                               makeAdd(rRange.lower_[i], rRange.len_[i]), "",
                               false, fill);
            }
        } else {
            fill = makeStmtSeq("", {});
        }

        Stmt flush;
        if (wRange_.count(newDef_)) {
            auto &&wRange = wRange_.at(newDef_);
            flush = makeStore("", oldVar_, indices, makeLoad(newVar_, indices));
            flushStmt_ = flush->id();
            if (idx1d.isValid()) {
                flush = makeIf("", makeLT(idx1d, def_->sizeLim_), flush);
            }
            for (int i = nDim - 1; i >= 0; i--) {
                flush = makeFor("", iters[i], wRange.lower_[i],
                                makeAdd(wRange.lower_[i], wRange.len_[i]), "",
                                false, flush);
            }
        } else {
            flush = makeStmtSeq("", {});
        }

        op = makeStmtSeq("", {fill, op, flush});
    }
    return op;
}

Stmt MakeFillAndFlush::visit(const VarDef &op) {
    if (op->id() == oldDef_) {
        def_ = op;
        auto ret = Mutator::visit(op);
        def_ = nullptr;
        return ret;
    } else {
        return Mutator::visit(op);
    }
}

Stmt MakeInitAndReduce::visitStmt(
    const Stmt &_op, const std::function<Stmt(const Stmt &)> &visitNode) {
    auto op = Mutator::visitStmt(_op, visitNode);
    if (op->id() == stmt_) {
        std::vector<std::string> iters;
        std::vector<Expr> indices;
        ASSERT(def_.isValid());
        int nDim = def_->buffer_->tensor().shape().size();
        iters.reserve(nDim);
        indices.reserve(nDim);
        for (int i = 0; i < nDim; i++) {
            std::string iter = "." + newVar_ + ".i" + std::to_string(i);
            indices.emplace_back(makeVar(iter));
            iters.emplace_back(std::move(iter));
        }

        Expr idx1d;
        if (def_->sizeLim_.isValid()) {
            auto &&shape = def_->buffer_->tensor().shape();
            for (int i = 0; i < nDim; i++) {
                idx1d = idx1d.isValid() ? makeMul(idx1d, shape[i]) : nullptr;
                idx1d =
                    idx1d.isValid() ? makeAdd(idx1d, indices[i]) : indices[i];
            }
        }

        if (!range_.count(newDef_)) {
            throw InvalidSchedule("No access to " + oldVar_ + " is found");
        }
        auto &&range = range_.at(newDef_);
        Stmt init = makeStore(
            "", newVar_, indices,
            makeNeutralVal(def_->buffer_->tensor().dtype(), reduce_->op_));
        initStmt_ = init->id();
        if (idx1d.isValid()) {
            init = makeIf("", makeLT(idx1d, def_->sizeLim_), init);
        }
        for (int i = nDim - 1; i >= 0; i--) {
            init = makeFor("", iters[i], range.lower_[i],
                           makeAdd(range.lower_[i], range.len_[i]), "", false,
                           init);
        }

        Stmt reduce = makeReduceTo("", oldVar_, indices, reduce_->op_,
                                   makeLoad(newVar_, indices), false);
        reduceStmt_ = reduce->id();
        if (idx1d.isValid()) {
            reduce = makeIf("", makeLT(idx1d, def_->sizeLim_), reduce);
        }
        for (int i = nDim - 1; i >= 0; i--) {
            reduce = makeFor("", iters[i], range.lower_[i],
                             makeAdd(range.lower_[i], range.len_[i]), "", false,
                             reduce);
        }

        op = makeStmtSeq("", {init, op, reduce});
    }
    return op;
}

Stmt MakeInitAndReduce::visit(const VarDef &op) {
    if (op->id() == oldDef_) {
        def_ = op;
        reduce_ = nullptr;
        auto ret = Mutator::visit(op);
        def_ = nullptr;
        return ret;
    } else if (op->id() == newDef_) {
        inNewVar_ = true;
        auto ret = Mutator::visit(op);
        inNewVar_ = false;
        return ret;
    } else {
        return Mutator::visit(op);
    }
}

Stmt MakeInitAndReduce::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    if (inNewVar_ && op->var_ == newVar_) {
        if (reduce_.isValid() && reduce_->op_ != op->op_) {
            throw InvalidSchedule(
                "Mixing different reduction operation is not allowed");
        }
        reduce_ = op;
    }
    return op;
}

Stmt MakeInitAndReduce::visit(const Store &op) {
    if (inNewVar_ && op->var_ == newVar_) {
        throw InvalidSchedule(
            "Any Store node in a cache_reduce region is not allowed");
    }
    return Mutator::visit(op);
}

Expr MakeInitAndReduce::visit(const Load &op) {
    if (inNewVar_ && op->var_ == newVar_) {
        throw InvalidSchedule(
            "Any Load node in a cache_reduce region is not allowed");
    }
    return Mutator::visit(op);
}

} // namespace ir
