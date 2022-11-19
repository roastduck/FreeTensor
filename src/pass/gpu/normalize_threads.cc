#ifdef FT_WITH_CUDA

#include <climits>

#include <analyze/all_uses.h>
#include <analyze/merge_no_deps_hint.h>
#include <pass/gpu/normalize_threads.h>
#include <pass/merge_and_hoist_if.h>
#include <pass/shrink_for.h>

namespace freetensor {

namespace gpu {

Expr NormalizeThreads::visit(const Var &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    if (varMap_.count(op->name_)) {
        auto &&info = varMap_.at(op->name_);
        return makeAdd(makeVar(info.newIter_), (*this)(info.offset_));
    }
    return op;
}

Stmt NormalizeThreads::makeParallelScopes(const Stmt &body) {
    auto zero = makeIntConst(0);
    auto one = makeIntConst(1);
    auto inf = makeIntConst(INT_MAX);
    Stmt ret = body;
    ret = makeFor(".threadIdx.x", zero, inf, one, inf,
                  Ref<ForProperty>::make()
                      ->withParallel(threadIdxX)
                      ->withNoDeps(mergeNoDepsHint(root_, loops_[threadIdxX])),
                  std::move(ret));
    ret = makeFor(".threadIdx.y", zero, inf, one, inf,
                  Ref<ForProperty>::make()
                      ->withParallel(threadIdxY)
                      ->withNoDeps(mergeNoDepsHint(root_, loops_[threadIdxY])),
                  std::move(ret));
    ret = makeFor(".threadIdx.z", zero, inf, one, inf,
                  Ref<ForProperty>::make()
                      ->withParallel(threadIdxZ)
                      ->withNoDeps(mergeNoDepsHint(root_, loops_[threadIdxZ])),
                  std::move(ret));
    ret = makeFor(".blockIdx.x", zero, inf, one, inf,
                  Ref<ForProperty>::make()->withParallel(blockIdxX)->withNoDeps(
                      mergeNoDepsHint(root_, loops_[blockIdxX])),
                  std::move(ret));
    ret = makeFor(".blockIdx.y", zero, inf, one, inf,
                  Ref<ForProperty>::make()->withParallel(blockIdxY)->withNoDeps(
                      mergeNoDepsHint(root_, loops_[blockIdxY])),
                  std::move(ret));
    ret = makeFor(".blockIdx.z", zero, inf, one, inf,
                  Ref<ForProperty>::make()->withParallel(blockIdxZ)->withNoDeps(
                      mergeNoDepsHint(root_, loops_[blockIdxZ])),
                  std::move(ret));
    return ret;
}

Stmt NormalizeThreads::doVisitStmt(const Stmt &_op) {
    Stmt op = _op;
    if (!inKernel_) {
        return op;
    }
    if (!inside_[threadIdxX]) {
        op = makeIf(makeEQ(makeVar(".threadIdx.x"), makeIntConst(0)), op);
    }
    if (!inside_[threadIdxY]) {
        op = makeIf(makeEQ(makeVar(".threadIdx.y"), makeIntConst(0)), op);
    }
    if (!inside_[threadIdxZ]) {
        op = makeIf(makeEQ(makeVar(".threadIdx.z"), makeIntConst(0)), op);
    }
    if (!inside_[blockIdxX]) {
        op = makeIf(makeEQ(makeVar(".blockIdx.x"), makeIntConst(0)), op);
    }
    if (!inside_[blockIdxY]) {
        op = makeIf(makeEQ(makeVar(".blockIdx.y"), makeIntConst(0)), op);
    }
    if (!inside_[blockIdxZ]) {
        op = makeIf(makeEQ(makeVar(".blockIdx.z"), makeIntConst(0)), op);
    }
    return op;
}

Stmt NormalizeThreads::doVisitFor(const For &_op) {
    if (std::holds_alternative<CUDAScope>(_op->property_->parallel_)) {
        auto newIter = "." + toString(_op->property_->parallel_);
        varMap_[_op->iter_] = {newIter, _op->begin_};
        inside_[_op->property_->parallel_]++;
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        varMap_.erase(_op->iter_);
        inside_[_op->property_->parallel_]--;
        loops_[_op->property_->parallel_].emplace_back(_op->id());
        return makeIf(makeLT(makeVar(newIter), op->len_), op->body_,
                      op->metadata(), op->id());
    } else {
        return Mutator::visit(_op);
    }
}

Stmt NormalizeThreads::visit(const For &op) {
    if (!inKernel_ &&
        std::holds_alternative<CUDAScope>(op->property_->parallel_)) {
        inKernel_ = true;
        auto ret = doVisitFor(op);
        inKernel_ = false;
        return makeParallelScopes(ret);
    } else {
        return doVisitFor(op);
    }
}

Stmt NormalizeThreads::visit(const VarDef &op) {
    if (!inKernel_) {
        switch (op->buffer_->mtype()) {
        case MemType::GPULocal:
        case MemType::GPUWarp:
        case MemType::GPUShared: {
            inKernel_ = true;
            auto ret = Mutator::visit(op);
            inKernel_ = false;
            return makeParallelScopes(ret);
        }
        default:
            return Mutator::visit(op);
        }
    } else {
        return Mutator::visit(op);
    }
}

Stmt NormalizeThreads::visit(const Store &op) {
    return doVisitStmt(Mutator::visit(op));
}

Stmt NormalizeThreads::visit(const ReduceTo &op) {
    return doVisitStmt(Mutator::visit(op));
}

Stmt NormalizeThreads::visit(const Eval &op) {
    return doVisitStmt(Mutator::visit(op));
}

bool CheckThreadNum::isLegalLen(const Expr &expr) {
    return isLegalLen(allNames(expr));
}

bool CheckThreadNum::isLegalLen(const std::unordered_set<std::string> &names) {
    for (auto &&name : names) {
        if (hasLoop(name)) {
            // Only iterators from outside of the kernel is OK
            if (openLoopsInKernel_.count(loop(name))) {
                return false;
            }
        } else if (!hasDef(name) || buffer(name)->mtype() != MemType::ByValue) {
            return false;
        }
    }
    return true;
}

Stmt CheckThreadNum::visit(const For &_op) {
    if (std::holds_alternative<CUDAScope>(_op->property_->parallel_)) {
        openLoopsInKernel_.insert(_op);

        auto oldInKernel = inKernel_;
        inKernel_ = true;
        auto __op = BaseClass::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        inKernel_ = oldInKernel;

        std::unordered_set<std::string> allLegalNames;
        for (auto &&name : names()) {
            if (isLegalLen({name}))
                allLegalNames.emplace(name);
        }

        if (!isLegalLen(op->begin_)) {
            op->body_ =
                makeIf(makeGE(makeVar(op->iter_), op->begin_), op->body_);
            Expr begin = bound_.getBound(op->begin_)
                             ->restrictScope(allLegalNames)
                             ->lowerExpr();
            if (!begin.isValid()) {
                throw InvalidProgram(
                    "Length of " + toString(op->property_->parallel_) +
                    " should have a finite bound. Note: if you are making a "
                    "dynamic ranged threadIdx or blockIdx loop, please use "
                    "memory type \"byvalue\" for its range, because it is used "
                    "both for launching the kernel and guarding the execution "
                    "inside the kernel");
            }
            op->begin_ = std::move(begin);
        }
        if (!isLegalLen(op->end_)) {
            op->body_ = makeIf(makeLT(makeVar(op->iter_), op->end_), op->body_);
            Expr end = bound_.getBound(op->end_)
                           ->restrictScope(allLegalNames)
                           ->upperExpr();
            if (!end.isValid()) {
                throw InvalidProgram(
                    "Length of " + toString(op->property_->parallel_) +
                    " should have a finite bound. Note: if you are making a "
                    "dynamic ranged threadIdx or blockIdx loop, please use "
                    "memory type \"byvalue\" for its range, because it is used "
                    "both for launching the kernel and guarding the execution "
                    "inside the kernel");
            }
            op->end_ = std::move(end);
        }
        ASSERT(op->step_->nodeType() == ASTNodeType::IntConst &&
               op->step_.as<IntConstNode>()->val_ == 1);
        op->len_ = makeSub(op->end_, op->begin_);

        openLoopsInKernel_.erase(_op);
        return op;
    } else {
        if (inKernel_) {
            openLoopsInKernel_.insert(_op);
            auto ret = BaseClass::visit(_op);
            openLoopsInKernel_.erase(_op);
            return ret;
        } else {
            return BaseClass::visit(_op);
        }
    }
}

Stmt normalizeThreads(const Stmt &_op) {
    auto op = NormalizeThreads(_op)(_op);
    op = shrinkFor(op);
    op = mergeAndHoistIf(op);
    op = CheckThreadNum()(op);
    return op;
}

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA
