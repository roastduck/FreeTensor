#include <algorithm>

#include <analyze/check_all_defined.h>
#include <analyze/comp_access_bound.h>
#include <math/min_max.h>

namespace freetensor {

static bool isSharedAmong(MemType mtype, const ParallelScope &parallel) {
    if (std::holds_alternative<CUDAScope>(parallel)) {
        if (std::get<CUDAScope>(parallel).level_ == CUDAScope::Thread) {
            switch (mtype) {
            case MemType::GPUGlobal:
            case MemType::GPUShared:
            case MemType::GPUWarp:
                return true;
            default:
                return false;
            }
        }
        if (std::get<CUDAScope>(parallel).level_ == CUDAScope::Block) {
            switch (mtype) {
            case MemType::GPUGlobal:
                return true;
            default:
                return false;
            }
        }
    }
    return false;
}

static bool isConstTrue(const Expr &expr) {
    return expr->nodeType() == ASTNodeType::BoolConst &&
           expr.as<BoolConstNode>()->val_;
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
        BaseClass::visit(op);
        defs_.erase(op->name_);
        return;
    }

    var_ = op->name_;
    defs_.insert(op->name_);
    BaseClass::visit(op);
    defs_.erase(op->name_);
    var_.clear();

    size_t n = op->buffer_->tensor()->shape().size();
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
            for (auto &&b : access_[j].lower_[i]) {
                if (checkAllDefined(defs_, b.allNames())) {
                    lowerItem.emplace_back(b.expr());
                }
            }
            lower.emplace_back(std::move(lowerItem));
        }

        for (size_t j = 0, jEnd = access_.size(); j < jEnd; j++) {
            ASSERT(access_[j].indices_.size() == n);
            auto &&index = access_[j].indices_[i];
            std::vector<Expr> upperItem(
                {makeSub(op->buffer_->tensor()->shape()[i], makeIntConst(1))});
            if (checkAllDefined(defs_, index)) {
                upperItem.emplace_back(index);
            }
            for (auto &&b : access_[j].upper_[i]) {
                if (checkAllDefined(defs_, b.allNames())) {
                    upperItem.emplace_back(b.expr());
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
        for (size_t i = conds().size(), iEnd = item.conds_.size(); i < iEnd;
             i++) {
            auto &&cond = item.conds_[i];
            if (!isConstTrue(cond) && checkAllDefined(defs_, cond)) {
                part = part.isValid() ? makeLAnd(part, cond) : cond;
            }
        }
        if (part.isValid()) {
            result_.cond_ =
                result_.cond_.isValid() ? makeLOr(result_.cond_, part) : part;
        } else {
            result_.cond_ = makeBoolConst(true);
        }
    }
}

void CompAccessBound::visit(const Load &op) {
    BaseClass::visit(op);
    if (op->var_ == var_ && mode_ & COMP_ACCESS_BOUND_READ) {
        access_.emplace_back(unique_, op->indices_, conds());
    }
}

void CompAccessBound::visit(const Store &op) {
    BaseClass::visit(op);
    if (op->var_ == var_ && mode_ & COMP_ACCESS_BOUND_WRITE) {
        access_.emplace_back(unique_, op->indices_, conds());
    }
}

void CompAccessBound::visit(const ReduceTo &op) {
    BaseClass::visit(op);
    if (op->var_ == var_) {
        access_.emplace_back(unique_, op->indices_, conds());
    }
}

void CompAccessBound::visit(const For &op) {
    if (isSharedAmong(mtype_, op->property_->parallel_)) {
        BaseClass::visit(op);
    } else {
        defs_.insert(op->iter_);
        BaseClass::visit(op);
        defs_.erase(op->iter_);
    }
}

AccessBound compAccessBound(const Stmt &op, const ID &varDefId,
                            CompAccessBoundMode mode) {
    FindMemType finder(varDefId);
    finder(op);
    CompAccessBound visitor(varDefId, finder.mtype(), mode);
    visitor(op);
    return visitor.result();
}

} // namespace freetensor
