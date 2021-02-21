#include <analyze/check_all_defined.h>
#include <analyze/comp_for_bound.h>

namespace ir {

Expr CompForBound::reduceMin(const Expr &reduction, const Expr &item) {
    return reduction.isValid() ? makeMin(reduction, item) : item;
}

Expr CompForBound::reduceMax(const Expr &reduction, const Expr &item) {
    return reduction.isValid() ? makeMax(reduction, item) : item;
}

Expr CompForBound::visit(const Var &op) {
    if (!inCond_) {
        uses_[op->name_].emplace_back(op); // use the old object
    }
    return Mutator::visit(op);
}

Stmt CompForBound::visit(const If &op) {
    inCond_ = true;
    auto cond = (*this)(op->cond_);
    inCond_ = false;
    auto thenCase = (*this)(op->thenCase_);
    auto elseCase = op->elseCase_.isValid() ? (*this)(op->elseCase_) : nullptr;
    return makeIf(op->id(), std::move(cond), std::move(thenCase),
                  std::move(elseCase));
}

Stmt CompForBound::visit(const For &_op) {
    uses_.erase(_op->iter_);

    defs_.insert(_op->iter_);
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    defs_.erase(_op->iter_);

    if (uses_.count(op->iter_)) {
        Expr lower, upper;
        for (auto &&use : uses_.at(op->iter_)) {
            if (lower_.count(use)) {
                Expr lowerItem;
                for (auto &&item : lower_.at(use)) {
                    if (checkAllDefined(defs_, item.expr_)) {
                        lowerItem = reduceMax(lowerItem, item.expr_);
                    }
                }
                lower = reduceMin(lower, lowerItem);
            } else {
                lower = op->begin_;
                break;
            }
        }

        for (auto &&use : uses_.at(op->iter_)) {
            if (upper_.count(use)) {
                Expr upperItem;
                for (auto &&item : upper_.at(use)) {
                    if (checkAllDefined(defs_, item.expr_)) {
                        upperItem = reduceMin(upperItem, item.expr_);
                    }
                }
                upper = reduceMax(upper, upperItem);
            } else {
                upper = makeSub(op->end_, makeIntConst(1));
                break;
            }
        }

        op->infoMaxBegin_ = lower;
        op->infoMinEnd_ =
            upper.isValid() ? makeAdd(upper, makeIntConst(1)) : nullptr;
    } else {
        // If the iterator is not used in any non-condition expressions, we
        // assume the loop can only run for 1 iteration. This may not be true in
        // some programs: E.g. `for i = 0 to 5 { a += b; }`. (FIXME)
        op->infoMaxBegin_ = op->begin_;
        op->infoMinEnd_ = makeAdd(op->begin_, makeIntConst(1));
    }
    return op;
}

Stmt CompForBound::visit(const VarDef &op) {
    defs_.insert(op->name_);
    auto ret = Mutator::visit(op);
    defs_.erase(op->name_);
    return ret;
}

} // namespace ir

