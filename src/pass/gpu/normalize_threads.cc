#ifdef FT_WITH_CUDA

#include <climits>

#include <analyze/merge_no_deps_hint.h>
#include <pass/gpu/normalize_thread_dims.h>
#include <pass/gpu/normalize_threads.h>
#include <pass/normalize_loops.h>
#include <pass/shrink_for.h>

namespace freetensor {

namespace gpu {

Expr NormalizeThreads::visit(const Var &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    if (varMap_.count(op->name_)) {
        return makeVar(varMap_.at(op->name_));
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
        varMap_[_op->iter_] = newIter;
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

Stmt normalizeThreads(const Stmt &_op) {
    auto op = normalizeLoops(_op, [](const For &l) {
        return std::holds_alternative<CUDAScope>(l->property_->parallel_);
    });
    op = NormalizeThreads(op)(op);
    op = shrinkFor(op);
    op = normalizeThreadDims(op);
    // NOTE: Although we have inserted a lot of identical `if`s, we must delay
    // `pass/merge_and_hoist_if` until we have done `pass/gpu/make_sync`.
    // Otherwise, we are introducing dependences between an `if`'s "then" case
    // and its "else" case, which is ill-defined in our IR.
    return op;
}

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA
