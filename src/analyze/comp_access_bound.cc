#include <analyze/check_all_defined.h>
#include <analyze/comp_access_bound.h>
#include <math/min_max.h>

namespace ir {

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
        std::vector<std::vector<Expr>> lower, upper;
        for (size_t j = 0, jEnd = access.size(); j < jEnd; j++) {
            ASSERT(access[j].indices_.size() == n);
            auto &&index = access[j].indices_[i];
            std::vector<Expr> lowerItem({makeIntConst(0)});
            if (checkAllDefined(defs_, index)) {
                lowerItem.emplace_back(index);
            }
            if (lower_.count(index)) {
                for (auto item : lower_.at(index)) {
                    if (checkAllDefined(defs_, item.expr())) {
                        lowerItem.emplace_back(item.expr());
                    }
                }
            }
            lower.emplace_back(std::move(lowerItem));
        }

        for (size_t j = 0, jEnd = access.size(); j < jEnd; j++) {
            ASSERT(access[j].indices_.size() == n);
            auto &&index = access[j].indices_[i];
            std::vector<Expr> upperItem(
                {makeSub(op->buffer_->tensor().shape()[i], makeIntConst(1))});
            if (checkAllDefined(defs_, index)) {
                upperItem.emplace_back(index);
            }
            if (upper_.count(index)) {
                for (auto item : upper_.at(index)) {
                    if (checkAllDefined(defs_, item.expr())) {
                        upperItem.emplace_back(item.expr());
                    }
                }
            }
            upper.emplace_back(std::move(upperItem));
        }

        auto l = makeMinMax(lower);
        auto u = makeMaxMin(upper);
        result.lower_.emplace_back(l);
        result.len_.emplace_back(makeAdd(makeSub(u, l), makeIntConst(1)));
    }

    for (auto &&item : access) {
        for (auto &&cond : item.conds_) {
            if (checkAllDefined(defs_, cond)) {
                result.cond_ = result.cond_.isValid()
                                   ? makeLAnd(result.cond_, cond)
                                   : cond;
            }
        }
    }

    results_[op->id()] = std::move(result);
}

void CompAccessBound::visit(const Load &op) {
    Visitor::visit(op);
    if (mode_ & COMP_ACCESS_BOUND_READ) {
        access_[op->var_].emplace_back(
            std::vector<Expr>(op->indices_.begin(), op->indices_.end()),
            condStack_);
    }
}

void CompAccessBound::visit(const Store &op) {
    Visitor::visit(op);
    if (mode_ & COMP_ACCESS_BOUND_WRITE) {
        access_[op->var_].emplace_back(
            std::vector<Expr>(op->indices_.begin(), op->indices_.end()),
            condStack_);
    }
}

void CompAccessBound::visit(const ReduceTo &op) {
    Visitor::visit(op);
    access_[op->var_].emplace_back(
        std::vector<Expr>(op->indices_.begin(), op->indices_.end()),
        condStack_);
}

void CompAccessBound::visit(const For &op) {
    defs_.insert(op->iter_);
    Visitor::visit(op);
    defs_.erase(op->iter_);
}

void CompAccessBound::visit(const If &op) {
    (*this)(op->cond_);
    condStack_.emplace_back(op->cond_);
    (*this)(op->thenCase_);
    if (op->elseCase_.isValid()) {
        condStack_.back() = makeLNot(op->cond_);
        (*this)(op->elseCase_);
    }
    condStack_.pop_back();
}

} // namespace ir

