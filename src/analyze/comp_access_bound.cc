#include <analyze/check_all_defined.h>
#include <analyze/comp_access_bound.h>
#include <math/min_max.h>

namespace ir {

static bool isSharedAmong(MemType mtype, const std::string &parallel) {
    if (parallel == "threadIdx.x" || parallel == "threadIdx.y" ||
        parallel == "threadIdx.z") {
        switch (mtype) {
        case MemType::GPUGlobal:
        case MemType::GPUShared:
            return true;
        default:
            return false;
        }
    }
    if (parallel == "blockIdx.x" || parallel == "blockIdx.y" ||
        parallel == "blockIdx.z") {
        switch (mtype) {
        case MemType::GPUGlobal:
            return true;
        default:
            return false;
        }
    }
    return false;
}

void FindMemType::visit(const VarDef &op) {
    Visitor::visit(op);
    if (op->id() == varDefId_) {
        mtype_ = op->buffer_->mtype();
    }
}

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
        Expr part;
        for (auto &&cond : item.conds_) {
            if (checkAllDefined(defs_, cond)) {
                part = part.isValid() ? makeLAnd(part, cond) : cond;
            }
        }
        if (part.isValid()) {
            result_.cond_ =
                result_.cond_.isValid() ? makeLOr(result_.cond_, part) : part;
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
    if (isSharedAmong(mtype_, op->property_.parallel_)) {
        Visitor::visit(op);
    } else {
        defs_.insert(op->iter_);
        Visitor::visit(op);
        defs_.erase(op->iter_);
    }
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

AccessBound
compAccessBound(const Stmt &op, const std::string &varDefId,
                const std::unordered_map<Expr, std::vector<LowerBound>> &lower,
                const std::unordered_map<Expr, std::vector<UpperBound>> &upper,
                CompAccessBoundMode mode) {
    FindMemType finder(varDefId);
    finder(op);
    CompAccessBound visitor(varDefId, finder.mtype(), lower, upper, mode);
    visitor(op);
    return visitor.result();
}

} // namespace ir
