#include <climits>

#include <pass/gpu/normalize_threads.h>
#include <pass/merge_and_hoist_if.h>
#include <pass/shrink_for.h>

namespace ir {

namespace gpu {

static std::vector<std::string> intersect(const std::vector<std::string> &lhs,
                                          const std::vector<std::string> &rhs) {
    std::vector<std::string> ret;
    for (auto &&item : lhs) {
        if (std::find(rhs.begin(), rhs.end(), item) != rhs.end()) {
            ret.emplace_back(item);
        }
    }
    return ret;
}

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
    if (!inside_["threadIdx.x"]) {
        op = makeIf("", makeEQ(makeVar(".threadIdx.x"), makeIntConst(0)), op);
    }
    if (!inside_["threadIdx.y"]) {
        op = makeIf("", makeEQ(makeVar(".threadIdx.y"), makeIntConst(0)), op);
    }
    if (!inside_["threadIdx.z"]) {
        op = makeIf("", makeEQ(makeVar(".threadIdx.z"), makeIntConst(0)), op);
    }
    if (!inside_["blockIdx.x"]) {
        op = makeIf("", makeEQ(makeVar(".blockIdx.x"), makeIntConst(0)), op);
    }
    if (!inside_["blockIdx.y"]) {
        op = makeIf("", makeEQ(makeVar(".blockIdx.y"), makeIntConst(0)), op);
    }
    if (!inside_["blockIdx.z"]) {
        op = makeIf("", makeEQ(makeVar(".blockIdx.z"), makeIntConst(0)), op);
    }
    return op;
}

Stmt NormalizeThreads::doVisitFor(const For &_op) {
    if (_op->property_.parallel_ == "blockIdx.x" ||
        _op->property_.parallel_ == "blockIdx.y" ||
        _op->property_.parallel_ == "blockIdx.z" ||
        _op->property_.parallel_ == "threadIdx.x" ||
        _op->property_.parallel_ == "threadIdx.y" ||
        _op->property_.parallel_ == "threadIdx.z") {
        auto newIter = "." + _op->property_.parallel_;
        varMap_[_op->iter_] = {newIter, _op->begin_};
        inside_[_op->property_.parallel_]++;
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        varMap_.erase(_op->iter_);
        inside_[_op->property_.parallel_]--;
        if (!noDeps_.count(_op->property_.parallel_)) {
            noDeps_[_op->property_.parallel_] = _op->property_.noDeps_;
        } else {
            noDeps_[_op->property_.parallel_] = intersect(
                noDeps_.at(_op->property_.parallel_), _op->property_.noDeps_);
        }
        return makeIf(op->id(), makeLT(makeVar(newIter), op->len_), op->body_);
    } else {
        return Mutator::visit(_op);
    }
}

Stmt NormalizeThreads::visit(const For &op) {
    if (!inKernel_ && (op->property_.parallel_ == "blockIdx.x" ||
                       op->property_.parallel_ == "blockIdx.y" ||
                       op->property_.parallel_ == "blockIdx.z" ||
                       op->property_.parallel_ == "threadIdx.x" ||
                       op->property_.parallel_ == "threadIdx.y" ||
                       op->property_.parallel_ == "threadIdx.z")) {
        inKernel_ = true;
        auto ret = doVisitFor(op);
        inKernel_ = false;
        auto zero = makeIntConst(0);
        auto one = makeIntConst(1);
        auto inf = makeIntConst(INT_MAX);
        ret = makeFor("", ".threadIdx.x", zero, inf, one, inf,
                      ForProperty()
                          .withParallel("threadIdx.x")
                          .withNoDeps(noDeps_["threadIdx.x"]),
                      ret);
        ret = makeFor("", ".threadIdx.y", zero, inf, one, inf,
                      ForProperty()
                          .withParallel("threadIdx.y")
                          .withNoDeps(noDeps_["threadIdx.y"]),
                      ret);
        ret = makeFor("", ".threadIdx.z", zero, inf, one, inf,
                      ForProperty()
                          .withParallel("threadIdx.z")
                          .withNoDeps(noDeps_["threadIdx.z"]),
                      ret);
        ret = makeFor("", ".blockIdx.x", zero, inf, one, inf,
                      ForProperty()
                          .withParallel("blockIdx.x")
                          .withNoDeps(noDeps_["blockIdx.x"]),
                      ret);
        ret = makeFor("", ".blockIdx.y", zero, inf, one, inf,
                      ForProperty()
                          .withParallel("blockIdx.y")
                          .withNoDeps(noDeps_["blockIdx.y"]),
                      ret);
        ret = makeFor("", ".blockIdx.z", zero, inf, one, inf,
                      ForProperty()
                          .withParallel("blockIdx.z")
                          .withNoDeps(noDeps_["blockIdx.z"]),
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

    if (!op->property_.parallel_.empty()) {
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
            throw InvalidProgram("Length of " + op->property_.parallel_ +
                                 " should have a finite bound");
        }
    }

    return op;
}

Stmt normalizeThreads(const Stmt &_op) {
    auto op = NormalizeThreads()(_op);
    op = shrinkFor(op);
    op = mergeAndHoistIf(op);
    op = CheckThreadNum()(op);
    return op;
}

} // namespace gpu

} // namespace ir
