#include <analyze/check_all_defined.h>
#include <analyze/comp_access_bound.h>

namespace ir {

Expr CompAccessBound::reduceMin(const Expr &reduction, const Expr &item) {
    return reduction.isValid() ? makeMin(reduction, item) : item;
}

Expr CompAccessBound::reduceMax(const Expr &reduction, const Expr &item) {
    return reduction.isValid() ? makeMax(reduction, item) : item;
}

void CompAccessBound::visit(const VarDef &op) {
    access_.erase(op->name_);

    defs_.insert(op->name_);
    Visitor::visit(op);
    defs_.erase(op->name_);

    if (op->buffer_->atype() != AccessType::Cache) {
        return;
    }

    if (!access_.count(op->name_)) {
        return;
    }
    auto &&access = access_.at(op->name_);

    size_t n = op->buffer_->tensor().shape().size();
    AccessBound result;
    result.lower_.reserve(n);
    result.len_.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Expr lower, upper;
        for (size_t j = 0, jEnd = access.size(); j < jEnd; j++) {
            ASSERT(access[j].size() == n);
            auto &&index = access[j][i];
            Expr lowerItem;
            if (checkAllDefined(defs_, index)) {
                lowerItem = index;
            }
            if (lower_.count(index)) {
                for (auto &&item : lower_.at(index)) {
                    if (checkAllDefined(defs_, item.expr_)) {
                        lowerItem = reduceMax(lowerItem, item.expr_);
                    }
                }
            }
            if (lowerItem.isValid()) {
                lower = reduceMin(lower, lowerItem);
            } else {
                lower = makeIntConst(0);
                break;
            }
        }

        for (size_t j = 0, jEnd = access.size(); j < jEnd; j++) {
            ASSERT(access[j].size() == n);
            auto &&index = access[j][i];
            Expr upperItem;
            if (checkAllDefined(defs_, index)) {
                upperItem = index;
            }
            if (upper_.count(index)) {
                for (auto &&item : upper_.at(index)) {
                    if (checkAllDefined(defs_, item.expr_)) {
                        upperItem = reduceMin(upperItem, item.expr_);
                    }
                }
            }
            if (upperItem.isValid()) {
                upper = reduceMax(upper, upperItem);
            } else {
                upper = op->buffer_->tensor().shape()[i];
                break;
            }
        }

        result.lower_.emplace_back(lower);
        result.len_.emplace_back(
            makeAdd(makeSub(upper, lower), makeIntConst(1)));
    }
    results_[op->id()] = std::move(result);
}

void CompAccessBound::visit(const Load &op) {
    Visitor::visit(op);
    if (mode_ & COMP_ACCESS_BOUND_READ) {
        access_[op->var_].emplace_back(op->indices_);
    }
}

void CompAccessBound::visit(const Store &op) {
    Visitor::visit(op);
    if (mode_ & COMP_ACCESS_BOUND_WRITE) {
        access_[op->var_].emplace_back(op->indices_);
    }
}

void CompAccessBound::visit(const ReduceTo &op) {
    Visitor::visit(op);
    access_[op->var_].emplace_back(op->indices_);
}

void CompAccessBound::visit(const For &op) {
    defs_.insert(op->iter_);
    Visitor::visit(op);
    defs_.erase(op->iter_);
}

} // namespace ir

