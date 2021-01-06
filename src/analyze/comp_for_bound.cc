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
            if (lower_.count(use.get())) {
                Expr lowerItem;
                for (auto &&item : lower_.at(use.get())) {
                    if (checkAllDefined(defs_, item)) {
                        lowerItem = reduceMax(lowerItem, item);
                    }
                }
                lower = reduceMin(lower, lowerItem);
            } else {
                lower = op->begin_;
                break;
            }
        }

        for (auto &&use : uses_.at(op->iter_)) {
            if (upper_.count(use.get())) {
                Expr upperItem;
                for (auto &&item : upper_.at(use.get())) {
                    if (checkAllDefined(defs_, item)) {
                        upperItem = reduceMin(upperItem, item);
                    }
                }
                upper = reduceMax(upper, upperItem);
            } else {
                upper = makeSub(op->end_, makeIntConst(1));
                break;
            }
        }

        op->info_max_begin_ = lower;
        op->info_min_end_ = makeAdd(upper, makeIntConst(1));
    } else {
        return op->body_;
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

