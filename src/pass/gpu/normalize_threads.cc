#include <climits>

#include <analyze/merge_no_deps_hint.h>
#include <pass/gpu/normalize_threads.h>
#include <pass/merge_and_hoist_if.h>
#include <pass/shrink_for.h>

namespace ir {

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

Stmt NormalizeThreads::doVisitStmt(const Stmt &_op) {
    Stmt op = _op;
    if (!inKernel_) {
        return op;
    }
    if (!inside_[threadIdxX]) {
        op = makeIf("", makeEQ(makeVar(".threadIdx.x"), makeIntConst(0)), op);
    }
    if (!inside_[threadIdxY]) {
        op = makeIf("", makeEQ(makeVar(".threadIdx.y"), makeIntConst(0)), op);
    }
    if (!inside_[threadIdxZ]) {
        op = makeIf("", makeEQ(makeVar(".threadIdx.z"), makeIntConst(0)), op);
    }
    if (!inside_[blockIdxX]) {
        op = makeIf("", makeEQ(makeVar(".blockIdx.x"), makeIntConst(0)), op);
    }
    if (!inside_[blockIdxY]) {
        op = makeIf("", makeEQ(makeVar(".blockIdx.y"), makeIntConst(0)), op);
    }
    if (!inside_[blockIdxZ]) {
        op = makeIf("", makeEQ(makeVar(".blockIdx.z"), makeIntConst(0)), op);
    }
    return op;
}

Stmt NormalizeThreads::doVisitFor(const For &_op) {
    if (std::holds_alternative<CUDAScope>(_op->property_.parallel_)) {
        auto newIter = "." + toString(_op->property_.parallel_);
        varMap_[_op->iter_] = {newIter, _op->begin_};
        inside_[_op->property_.parallel_]++;
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        varMap_.erase(_op->iter_);
        inside_[_op->property_.parallel_]--;
        loops_[_op->property_.parallel_].emplace_back(_op->id());
        return makeIf(op->id(), makeLT(makeVar(newIter), op->len_), op->body_);
    } else {
        return Mutator::visit(_op);
    }
}

Stmt NormalizeThreads::visit(const For &op) {
    if (!inKernel_ &&
        std::holds_alternative<CUDAScope>(op->property_.parallel_)) {
        inKernel_ = true;
        auto ret = doVisitFor(op);
        inKernel_ = false;
        auto zero = makeIntConst(0);
        auto one = makeIntConst(1);
        auto inf = makeIntConst(INT_MAX);
        ret =
            makeFor("", ".threadIdx.x", zero, inf, one, inf,
                    ForProperty()
                        .withParallel(threadIdxX)
                        .withNoDeps(mergeNoDepsHint(root_, loops_[threadIdxX])),
                    ret);
        ret =
            makeFor("", ".threadIdx.y", zero, inf, one, inf,
                    ForProperty()
                        .withParallel(threadIdxY)
                        .withNoDeps(mergeNoDepsHint(root_, loops_[threadIdxY])),
                    ret);
        ret =
            makeFor("", ".threadIdx.z", zero, inf, one, inf,
                    ForProperty()
                        .withParallel(threadIdxZ)
                        .withNoDeps(mergeNoDepsHint(root_, loops_[threadIdxZ])),
                    ret);
        ret = makeFor("", ".blockIdx.x", zero, inf, one, inf,
                      ForProperty().withParallel(blockIdxX).withNoDeps(
                          mergeNoDepsHint(root_, loops_[blockIdxX])),
                      ret);
        ret = makeFor("", ".blockIdx.y", zero, inf, one, inf,
                      ForProperty().withParallel(blockIdxY).withNoDeps(
                          mergeNoDepsHint(root_, loops_[blockIdxY])),
                      ret);
        ret = makeFor("", ".blockIdx.z", zero, inf, one, inf,
                      ForProperty().withParallel(blockIdxZ).withNoDeps(
                          mergeNoDepsHint(root_, loops_[blockIdxZ])),
                      ret);
        return ret;
    } else {
        return doVisitFor(op);
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

Stmt CheckThreadNum::visit(const For &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();

    if (op->property_.parallel_ != serialScope) {
        if (op->begin_->nodeType() != ASTNodeType::IntConst) {
            op->body_ =
                makeIf("", makeGE(makeVar(op->iter_), op->begin_), op->body_);
            op->begin_ = makeIntConst(bound_.getIntLower(op->begin_));
        }
        if (op->end_->nodeType() != ASTNodeType::IntConst) {
            op->body_ =
                makeIf("", makeLT(makeVar(op->iter_), op->end_), op->body_);
            op->end_ = makeIntConst(bound_.getIntUpper(op->end_));
        }
        ASSERT(op->begin_->nodeType() == ASTNodeType::IntConst);
        ASSERT(op->end_->nodeType() == ASTNodeType::IntConst);
        op->len_ = makeIntConst(op->end_.as<IntConstNode>()->val_ -
                                op->begin_.as<IntConstNode>()->val_);
        if (op->end_.as<IntConstNode>()->val_ == INT_MAX) {
            throw InvalidProgram("Length of " +
                                 toString(op->property_.parallel_) +
                                 " should have a finite bound");
        }
    }

    return op;
}

Stmt normalizeThreads(const Stmt &_op) {
    auto op = NormalizeThreads(_op)(_op);
    op = shrinkFor(op);
    op = mergeAndHoistIf(op);
    op = CheckThreadNum()(op);
    return op;
}

} // namespace gpu

} // namespace ir
