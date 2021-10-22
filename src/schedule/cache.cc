#include <climits>
#include <cmath>

#include <pass/make_reduction.h>
#include <pass/remove_writes.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <schedule/cache.h>
#include <schedule/check_var_cross_parallel.h>

namespace ir {

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
                         std::move(ret), false);
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
        fill = makeStore("", newVar_, indices, makeLoad(oldVar_, indices));
        fillStmt_ = fill->id();
        if (idx1d.isValid()) {
            fill = makeIf("", makeLT(idx1d, def_->sizeLim_), fill);
        }
        for (int i = nDim - 1; i >= 0; i--) {
            fill = makeFor("", iters[i], rRange_.lower_[i],
                           makeAdd(rRange_.lower_[i], rRange_.len_[i]),
                           rRange_.len_[i], false, ForProperty(), fill);
        }
        if (rRange_.cond_.isValid()) {
            fill = makeIf("", rRange_.cond_, fill);
        }

        Stmt flush;
        flush = makeStore("", oldVar_, indices, makeLoad(newVar_, indices));
        flushStmt_ = flush->id();
        if (idx1d.isValid()) {
            flush = makeIf("", makeLT(idx1d, def_->sizeLim_), flush);
        }
        for (int i = nDim - 1; i >= 0; i--) {
            flush = makeFor("", iters[i], wRange_.lower_[i],
                            makeAdd(wRange_.lower_[i], wRange_.len_[i]),
                            wRange_.len_[i], false, ForProperty(), flush);
        }
        if (wRange_.cond_.isValid()) {
            flush = makeIf("", wRange_.cond_, flush);
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

        Stmt init = makeStore(
            "", newVar_, indices,
            neutralVal(def_->buffer_->tensor().dtype(), reduce_->op_));
        initStmt_ = init->id();
        if (idx1d.isValid()) {
            init = makeIf("", makeLT(idx1d, def_->sizeLim_), init);
        }
        for (int i = nDim - 1; i >= 0; i--) {
            init = makeFor("", iters[i], range_.lower_[i],
                           makeAdd(range_.lower_[i], range_.len_[i]),
                           range_.len_[i], false, ForProperty(), init);
        }

        Stmt reduce = makeReduceTo("", oldVar_, indices, reduce_->op_,
                                   makeLoad(newVar_, indices), false);
        reduceStmt_ = reduce->id();
        if (idx1d.isValid()) {
            reduce = makeIf("", makeLT(idx1d, def_->sizeLim_), reduce);
        }
        for (int i = nDim - 1; i >= 0; i--) {
            reduce = makeFor("", iters[i], range_.lower_[i],
                             makeAdd(range_.lower_[i], range_.len_[i]),
                             range_.len_[i], false, ForProperty(), reduce);
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

std::pair<Stmt, std::tuple<std::string, std::string, std::string, std::string>>
cache(const Stmt &_ast, const std::string &stmt, const std::string &var,
      MemType mtype) {
    std::string fillStmt, flushStmt, newVar, oldDef, newDef;
    MakeCacheVar makeCacheVar(stmt, var, mtype, false);
    auto ast = makeCacheVar(_ast);
    newVar = makeCacheVar.newVar();
    oldDef = makeCacheVar.oldDef();
    newDef = makeCacheVar.newDef();
    if (newDef.empty()) {
        throw InvalidSchedule("Statement " + stmt + " not found");
    }

    BuiltinSimplify::LowerBoundsMap lower;
    BuiltinSimplify::UpperBoundsMap upper;
    std::tie(ast, lower, upper) = simplifyAndGetBounds<BuiltinSimplify>(ast);
    auto rBound =
        compAccessBound(ast, newDef, lower, upper, COMP_ACCESS_BOUND_READ);
    auto wBound =
        compAccessBound(ast, newDef, lower, upper, COMP_ACCESS_BOUND_WRITE);
    MakeFillAndFlush makeFillAndFlush(stmt, var, newVar, oldDef, rBound,
                                      wBound);
    ast = makeFillAndFlush(ast);
    fillStmt = makeFillAndFlush.fillStmt();
    flushStmt = makeFillAndFlush.flushStmt();

    ast = shrinkSingleVar(ast, newDef);
    ast = removeWrites(ast, newDef);
    checkVarCrossParallel(ast, newDef, mtype);
    return std::make_pair(
        ast, std::make_tuple(std::move(fillStmt), std::move(flushStmt),
                             std::move(newVar), std::move(newDef)));
}

std::pair<Stmt, std::tuple<std::string, std::string, std::string, std::string>>
cacheReduction(const Stmt &_ast, const std::string &stmt,
               const std::string &var, MemType mtype) {
    std::string initStmt, reduceStmt, newVar, oldDef, newDef;
    auto ast = makeReduction(_ast);

    MakeCacheVar makeCacheVar(stmt, var, mtype, true);
    ast = makeCacheVar(ast);
    newVar = makeCacheVar.newVar();
    oldDef = makeCacheVar.oldDef();
    newDef = makeCacheVar.newDef();
    if (newDef.empty()) {
        throw InvalidSchedule("Statement " + stmt + " not found");
    }

    BuiltinSimplify::LowerBoundsMap lower;
    BuiltinSimplify::UpperBoundsMap upper;
    std::tie(ast, lower, upper) = simplifyAndGetBounds<BuiltinSimplify>(ast);
    auto bound = compAccessBound(ast, newDef, lower, upper);
    MakeInitAndReduce makeInitAndReduce(stmt, var, newVar, oldDef, newDef,
                                        bound);
    ast = makeInitAndReduce(ast);
    initStmt = makeInitAndReduce.initStmt();
    reduceStmt = makeInitAndReduce.reduceStmt();

    ast = shrinkSingleVar(ast, newDef);
    ast = removeWrites(ast, newDef);
    checkVarCrossParallel(ast, newDef, mtype);
    return std::make_pair(
        ast, std::make_tuple(std::move(initStmt), std::move(reduceStmt),
                             std::move(newVar), std::move(newDef)));
}

} // namespace ir
