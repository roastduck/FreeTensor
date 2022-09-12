#include <climits>
#include <cmath>

#include <itertools.hpp>

#include <pass/make_nested_loops.h>
#include <pass/remove_writes.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <schedule/cache.h>
#include <schedule/check_var_cross_parallel.h>

namespace freetensor {

Stmt MakeCacheVar::visitStmt(const Stmt &op) {
    if (op->id() == stmt_) {
        if (!def_.isValid()) {
            throw InvalidSchedule("Variable " + oldVar_ + " not found");
        }
        inStmt_ = true;
        auto ret = Mutator::visitStmt(op);
        inStmt_ = false;
        Ref<Buffer> newBuffer =
            makeBuffer(def_->buffer_->tensor(), AccessType::Cache, mtype_);
        ret = makeVarDef(newVar_, std::move(newBuffer), nullptr, std::move(ret),
                         false);
        oldDef_ = def_->id();
        newDef_ = ret->id();
        return ret;
    } else {
        return Mutator::visitStmt(op);
    }
}

Stmt MakeCacheVar::visit(const VarDef &op) {
    if (op->name_ == oldVar_) {
        if (def_.isValid()) {
            throw InvalidProgram(
                "Nested VarDef with the same buffer name is not allowed");
        }
        def_ = op;
        auto ret = Mutator::visit(op);
        def_ = nullptr;
        return ret;
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

Stmt MakeFillAndFlush::visitStmt(const Stmt &_op) {
    auto op = Mutator::visitStmt(_op);
    if (op->id() == stmt_) {
        std::vector<Expr> indices;
        ASSERT(def_.isValid());
        int nDim = def_->buffer_->tensor()->shape().size();
        indices.reserve(nDim);
        for (int i = 0; i < nDim; i++) {
            std::string iter = "." + newVar_ + ".i" + std::to_string(i);
            indices.emplace_back(makeVar(iter));
        }

        Expr idx1d, sizeLim;
        if (def_->ioTensor_.isValid()) {
            for (auto &&[idx, dim] :
                 iter::zip(indices, def_->buffer_->tensor()->shape())) {
                idx1d = idx1d.isValid() ? makeMul(idx1d, dim) : nullptr;
                idx1d = idx1d.isValid() ? makeAdd(idx1d, idx) : idx;
            }
            for (Expr dim : def_->ioTensor_->shape()) {
                sizeLim = sizeLim.isValid() ? makeMul(sizeLim, dim) : dim;
            }
        }

        Stmt fill;
        fill = makeStore(
            newVar_, indices,
            makeLoad(oldVar_, indices, def_->buffer_->tensor()->dtype()));
        fillStmt_ = fill->id();
        if (idx1d.isValid()) {
            fill = makeIf(makeLT(idx1d, sizeLim), fill);
        }
        fill = makeNestedLoops(indices, rwRange_.lower_, iter::repeat(nullptr),
                               iter::repeat(makeIntConst(1)), rwRange_.len_,
                               iter::repeat(Ref<ForProperty>::make()), fill);
        if (rwRange_.cond_.isValid()) {
            fill = makeIf(rwRange_.cond_, fill);
        }

        Stmt flush;
        flush = makeStore(
            oldVar_, indices,
            makeLoad(newVar_, indices, def_->buffer_->tensor()->dtype()));
        flushStmt_ = flush->id();
        if (idx1d.isValid()) {
            flush = makeIf(makeLT(idx1d, sizeLim), flush);
        }
        flush = makeNestedLoops(indices, wRange_.lower_, iter::repeat(nullptr),
                                iter::repeat(makeIntConst(1)), wRange_.len_,
                                iter::repeat(Ref<ForProperty>::make()), flush);
        if (wRange_.cond_.isValid()) {
            flush = makeIf(wRange_.cond_, flush);
        }

        op = makeStmtSeq({fill, op, flush});
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

Stmt MakeInitAndReduce::visitStmt(const Stmt &_op) {
    auto op = Mutator::visitStmt(_op);
    if (op->id() == stmt_) {
        if (!reduce_.isValid()) {
            throw InvalidSchedule("The cached statement is not reducing into "
                                  "the specific variable");
        }
        std::vector<Expr> indices;
        ASSERT(def_.isValid());
        int nDim = def_->buffer_->tensor()->shape().size();
        indices.reserve(nDim);
        for (int i = 0; i < nDim; i++) {
            std::string iter = "." + newVar_ + ".i" + std::to_string(i);
            indices.emplace_back(makeVar(iter));
        }

        Expr idx1d, sizeLim;
        if (def_->ioTensor_.isValid()) {
            for (auto &&[idx, dim] :
                 iter::zip(indices, def_->buffer_->tensor()->shape())) {
                idx1d = idx1d.isValid() ? makeMul(idx1d, dim) : nullptr;
                idx1d = idx1d.isValid() ? makeAdd(idx1d, idx) : idx;
            }
            for (Expr dim : def_->ioTensor_->shape()) {
                sizeLim = sizeLim.isValid() ? makeMul(sizeLim, dim) : dim;
            }
        }

        Stmt init = makeStore(
            newVar_, indices,
            neutralVal(def_->buffer_->tensor()->dtype(), reduce_->op_));
        initStmt_ = init->id();
        if (idx1d.isValid()) {
            init = makeIf(makeLT(idx1d, sizeLim), init);
        }
        init = makeNestedLoops(indices, range_.lower_, iter::repeat(nullptr),
                               iter::repeat(makeIntConst(1)), range_.len_,
                               iter::repeat(Ref<ForProperty>::make()), init);

        Stmt reduce = makeReduceTo(
            oldVar_, indices, reduce_->op_,
            makeLoad(newVar_, indices, def_->buffer_->tensor()->dtype()),
            false);
        reduceStmt_ = reduce->id();
        if (idx1d.isValid()) {
            reduce = makeIf(makeLT(idx1d, sizeLim), reduce);
        }
        reduce =
            makeNestedLoops(indices, range_.lower_, iter::repeat(nullptr),
                            iter::repeat(makeIntConst(1)), range_.len_,
                            iter::repeat(Ref<ForProperty>::make()), reduce);

        op = makeStmtSeq({init, op, reduce});
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

std::pair<Stmt, std::tuple<ID, ID, std::string, ID>>
cache(const Stmt &_ast, const ID &stmt, const std::string &var, MemType mtype) {
    MakeCacheVar makeCacheVar(stmt, var, mtype, false);
    auto ast = makeCacheVar(_ast);
    auto newVar = makeCacheVar.newVar();
    auto oldDef = makeCacheVar.oldDef();
    auto newDef = makeCacheVar.newDef();
    if (!newDef.isValid()) {
        throw InvalidSchedule("Statement " + toString(stmt) + " not found");
    }

    ast = simplify(ast);
    auto rwBound = compAccessBound(ast, newDef);
    auto wBound = compAccessBound(ast, newDef, COMP_ACCESS_BOUND_WRITE);
    MakeFillAndFlush makeFillAndFlush(stmt, var, newVar, oldDef, rwBound,
                                      wBound);
    ast = makeFillAndFlush(ast);
    auto fillStmt = makeFillAndFlush.fillStmt();
    auto flushStmt = makeFillAndFlush.flushStmt();

    ast = simplify(ast);
    ast = shrinkSingleVar(ast, newDef);
    ast = removeWrites(ast, newDef);
    checkVarCrossParallel(ast, newDef, mtype);
    return {ast,
            {std::move(fillStmt), std::move(flushStmt), std::move(newVar),
             std::move(newDef)}};
}

std::pair<Stmt, std::tuple<ID, ID, std::string, ID>>
cacheReduction(const Stmt &_ast, const ID &stmt, const std::string &var,
               MemType mtype) {
    MakeCacheVar makeCacheVar(stmt, var, mtype, true);
    auto ast = makeCacheVar(_ast);
    auto newVar = makeCacheVar.newVar();
    auto oldDef = makeCacheVar.oldDef();
    auto newDef = makeCacheVar.newDef();
    if (!newDef.isValid()) {
        throw InvalidSchedule("Statement " + toString(stmt) + " not found");
    }

    ast = simplify(ast);
    auto bound = compAccessBound(ast, newDef);
    MakeInitAndReduce makeInitAndReduce(stmt, var, newVar, oldDef, newDef,
                                        bound);
    ast = makeInitAndReduce(ast);
    auto initStmt = makeInitAndReduce.initStmt();
    auto reduceStmt = makeInitAndReduce.reduceStmt();

    ast = simplify(ast);
    ast = shrinkSingleVar(ast, newDef);
    ast = removeWrites(ast, newDef);
    checkVarCrossParallel(ast, newDef, mtype);
    return {ast,
            {std::move(initStmt), std::move(reduceStmt), std::move(newVar),
             std::move(newDef)}};
}

} // namespace freetensor
