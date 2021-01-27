#include <climits>

#include <pass/gpu/normalize_threads.h>
#include <pass/merge_if.h>
#include <pass/shrink_for.h>

namespace ir {

namespace gpu {

Expr NormalizeThreads::visit(const Var &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    if (varMap_.count(op->name_)) {
        auto &&info = varMap_.at(op->name_);
        op = makeSub(makeVar(info.newIter_), info.offset_);
    }
    return op;
}

Stmt NormalizeThreads::doVisitStmt(const Stmt &_op) {
    Stmt op = _op;
    if (!inKernel_) {
        return op;
    }
    if (!inside_.count("threadIdx.x")) {
        op = makeIf("", makeEQ(makeVar(".threadIdx.x"), makeIntConst(0)), op);
    }
    if (!inside_.count("threadIdx.y")) {
        op = makeIf("", makeEQ(makeVar(".threadIdx.y"), makeIntConst(0)), op);
    }
    if (!inside_.count("threadIdx.z")) {
        op = makeIf("", makeEQ(makeVar(".threadIdx.z"), makeIntConst(0)), op);
    }
    if (!inside_.count("blockIdx.x")) {
        op = makeIf("", makeEQ(makeVar(".blockIdx.x"), makeIntConst(0)), op);
    }
    if (!inside_.count("blockIdx.y")) {
        op = makeIf("", makeEQ(makeVar(".blockIdx.y"), makeIntConst(0)), op);
    }
    if (!inside_.count("blockIdx.z")) {
        op = makeIf("", makeEQ(makeVar(".blockIdx.z"), makeIntConst(0)), op);
    }
    return op;
}

Stmt NormalizeThreads::doVisitFor(const For &_op) {
    if (_op->parallel_ == "blockIdx.x" || _op->parallel_ == "blockIdx.y" ||
        _op->parallel_ == "blockIdx.z" || _op->parallel_ == "threadIdx.x" ||
        _op->parallel_ == "threadIdx.y" || _op->parallel_ == "threadIdx.z") {
        auto newIter = "." + _op->parallel_;
        varMap_[_op->iter_] = {newIter, _op->begin_};
        inside_.insert(_op->parallel_);
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        varMap_.erase(_op->iter_);
        inside_.erase(_op->parallel_);
        return makeIf(op->id(),
                      makeLT(makeVar(newIter), makeSub(op->end_, op->begin_)),
                      op->body_);
    } else {
        return Mutator::visit(_op);
    }
}

Stmt NormalizeThreads::visit(const For &op) {
    if (!inKernel_ &&
        (op->parallel_ == "blockIdx.x" || op->parallel_ == "blockIdx.y" ||
         op->parallel_ == "blockIdx.z" || op->parallel_ == "threadIdx.x" ||
         op->parallel_ == "threadIdx.y" || op->parallel_ == "threadIdx.z")) {
        inKernel_ = true;
        auto ret = doVisitFor(op);
        inKernel_ = false;
        ret = makeFor("", ".threadIdx.x", makeIntConst(0),
                      makeIntConst(INT_MAX), "threadIdx.x", ret);
        ret = makeFor("", ".threadIdx.y", makeIntConst(0),
                      makeIntConst(INT_MAX), "threadIdx.y", ret);
        ret = makeFor("", ".threadIdx.z", makeIntConst(0),
                      makeIntConst(INT_MAX), "threadIdx.z", ret);
        ret = makeFor("", ".blockIdx.x", makeIntConst(0), makeIntConst(INT_MAX),
                      "blockIdx.x", ret);
        ret = makeFor("", ".blockIdx.y", makeIntConst(0), makeIntConst(INT_MAX),
                      "blockIdx.y", ret);
        ret = makeFor("", ".blockIdx.z", makeIntConst(0), makeIntConst(INT_MAX),
                      "blockIdx.z", ret);
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

Stmt normalizeThreads(const Stmt &_op) {
    auto op = NormalizeThreads()(_op);
    op = shrinkFor(op);
    op = mergeIf(op);
    // TODO: Check for unbounded loops
    return op;
}

} // namespace gpu

} // namespace ir

