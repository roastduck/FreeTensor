#include <analyze/check_all_defined.h>
#include <analyze/comp_access_bound.h>
#include <math/min_max.h>

namespace ir {

void CompAccessBound::visit(const VarDef &op) {
    if (op->id() != varDefId_) {
        defs_.insert(op->name_);
        Visitor::visit(op);
        defs_.erase(op->name_);
        return;
    }

    var_ = op->name_;
    defs_.insert(op->name_);
    Visitor::visit(op);
    defs_.erase(op->name_);
    var_.clear();

    size_t n = op->buffer_->tensor().shape().size();
    result_.lower_.reserve(n);
    result_.len_.reserve(n);

    if (access_.empty()) {
        result_.lower_.insert(result_.lower_.end(), n, makeIntConst(0));
        result_.len_.insert(result_.len_.end(), n, makeIntConst(0));
        result_.cond_ = makeBoolConst(false);
        return;
    }

    for (size_t i = 0; i < n; i++) {
        std::vector<std::vector<Expr>> lower, upper;
        for (size_t j = 0, jEnd = access_.size(); j < jEnd; j++) {
            ASSERT(access_[j].indices_.size() == n);
            auto &&index = access_[j].indices_[i];
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

        for (size_t j = 0, jEnd = access_.size(); j < jEnd; j++) {
            ASSERT(access_[j].indices_.size() == n);
            auto &&index = access_[j].indices_[i];
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
        result_.lower_.emplace_back(l);
        result_.len_.emplace_back(makeAdd(makeSub(u, l), makeIntConst(1)));
    }

    for (auto &&item : access_) {
        for (auto &&cond : item.conds_) {
            if (checkAllDefined(defs_, cond)) {
                result_.cond_ = result_.cond_.isValid()
                                    ? makeLAnd(result_.cond_, cond)
                                    : cond;
            }
        }
    }
}

void CompAccessBound::visit(const Load &op) {
    Visitor::visit(op);
    if (op->var_ == var_ && mode_ & COMP_ACCESS_BOUND_READ) {
        access_.emplace_back(
            std::vector<Expr>(op->indices_.begin(), op->indices_.end()),
            condStack_);
    }
}

void CompAccessBound::visit(const Store &op) {
    Visitor::visit(op);
    if (op->var_ == var_ && mode_ & COMP_ACCESS_BOUND_WRITE) {
        access_.emplace_back(
            std::vector<Expr>(op->indices_.begin(), op->indices_.end()),
            condStack_);
    }
}

void CompAccessBound::visit(const ReduceTo &op) {
    Visitor::visit(op);
    if (op->var_ == var_) {
        access_.emplace_back(
            std::vector<Expr>(op->indices_.begin(), op->indices_.end()),
            condStack_);
    }
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

