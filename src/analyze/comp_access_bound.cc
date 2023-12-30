#include <algorithm>

#include <analyze/check_all_defined.h>
#include <analyze/comp_access_bound.h>
#include <math/min_max.h>
#include <pass/pb_simplify.h>

namespace freetensor {

static bool isSharedAmong(MemType mtype, const ParallelScope &parallel) {
    if (std::holds_alternative<CUDAScope>(parallel)) {
        if (std::get<CUDAScope>(parallel).level_ == CUDAScope::Thread) {
            switch (mtype) {
            case MemType::GPUGlobal:
            case MemType::GPUGlobalHeap:
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
            case MemType::GPUGlobalHeap:
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

void CompAccessBound::visitStmt(const Stmt &stmt) {
    // CompUniqueBounds requires one instance per Stmt
    auto uniqueOfOuterStmt = unique_;
    unique_ = Ref<CompUniqueBoundsPB>::make(*this);

    if (stmt->id() == filterSubTree_) {
        filtered_ = true;
        BaseClass::visitStmt(stmt);
        filtered_ = false;
    } else {
        BaseClass::visitStmt(stmt);
    }

    unique_ = uniqueOfOuterStmt;
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
    defsAtVarDef_[op->name_] = defs_;
    BaseClass::visit(op);
    defsAtVarDef_.erase(op->name_);
    defs_.erase(op->name_);
    var_.clear();

    size_t n = op->buffer_->tensor()->shape().size();
    result_.lower_.reserve(n);
    result_.upper_.reserve(n);
    result_.len_.reserve(n);

    if (access_.empty()) {
        result_.lower_.insert(result_.lower_.end(), n, makeIntConst(0));
        result_.upper_.insert(result_.upper_.end(), n, makeIntConst(-1));
        result_.len_.insert(result_.len_.end(), n, makeIntConst(0));
        result_.cond_ = makeBoolConst(false);
        return;
    }

    for (size_t i = 0; i < n; i++) {
        // union the bounds of all accesses and get the lower and upper
        // expression
        auto [l, u] = unique_->unionBounds(
            // get bounds of the i-th dimension
            access_ | views::transform([&](auto &&a) { return a.bounds_[i]; }) |
            // ... and pack into vector
            ranges::to<std::vector>());
        // include the original trivial bounds, if specified
        if (includeTrivialBound_) {
            auto &&tl = makeIntConst(0);
            auto &&tu =
                makeSub(op->buffer_->tensor()->shape()[i], makeIntConst(1));
            l = l.isValid() ? makeMax(l, tl) : tl;
            u = u.isValid() ? makeMin(u, tu) : tu;
        }
        result_.lower_.emplace_back(l);
        result_.upper_.emplace_back(u);
        if (l.isValid() && u.isValid()) {
            auto diff = makeSub(u, l);

            // Suppose `upper = min(a, c)`, `lower = max(b, c)`, and we have no
            // knowledge about `a` or `b`, and all we know is we have `c` in
            // both `lower` and `upper`, we can simplify `len` to `1`. However,
            // directly analyzing the bounds of `upper - lower` only results in
            // its upper bound `upper - lower <= min(a - b, a - c, c - b, 0)`,
            // but no knowledge lower bound, so `pass/simplify` cannot do the
            // simplification. We explicitly mark `upper - lower >= 0` here by
            // `upper - lower = max(upper - lower, 0)`, to enable simplifying
            // `upper - lower` to 0.
            //
            // Note that this breaks the semantics and makes the length of a
            // dimension at least 1, instead of 0, and prohibits some "optional"
            // variables. However, this is actually beneficial, because a
            // 0-or-1-lengthed variable will end up in the heap (beacuase they
            // have "dynamic" length, and a 1-lengthed variable, although larger
            // by 1, will end up in registers
            diff = makeMax(diff, makeIntConst(0));

            result_.len_.emplace_back(makeAdd(diff, makeIntConst(1)));
        } else {
            result_.len_.emplace_back(nullptr);
        }
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
    if (filtered_ && op->var_ == var_ && mode_ & COMP_ACCESS_BOUND_READ) {
        access_.emplace_back(*unique_, op->indices_, conds(),
                             defsAtVarDef_.at(op->var_));
    }
}

void CompAccessBound::visit(const Store &op) {
    BaseClass::visit(op);
    if (filtered_ && op->var_ == var_ && mode_ & COMP_ACCESS_BOUND_WRITE) {
        access_.emplace_back(*unique_, op->indices_, conds(),
                             defsAtVarDef_.at(op->var_));
    }
}

void CompAccessBound::visit(const ReduceTo &op) {
    BaseClass::visit(op);
    if (filtered_ && op->var_ == var_) {
        access_.emplace_back(*unique_, op->indices_, conds(),
                             defsAtVarDef_.at(op->var_));
    }
}

void CompAccessBound::visit(const For &op) {
    if (isSharedAmong(mtype_, op->property_->parallel_)) {
        // Suppose the only access to tensor `t` is `t[i, ...]`, where `i` is a
        // parallel index (e.g. CUDA thread), we cannot shrink `t[i, ...]` to
        // `t[...]` too early before all schedules are done, or we are not able
        // to schedule a collaborative fetch. This does not apply to OpenMP
        // threads, because we cannot do a collaborative fetch anyway in OpenMP.
        BaseClass::visit(op);
    } else {
        defs_.insert(op->iter_);
        BaseClass::visit(op);
        defs_.erase(op->iter_);
    }
}

AccessBound compAccessBound(const Stmt &op, const ID &varDefId,
                            CompAccessBoundMode mode, bool includeTrivialBound,
                            const ID &filterSubTree) {
    FindMemType finder(varDefId);
    finder(op);
    CompAccessBound visitor(varDefId, finder.mtype(), mode, includeTrivialBound,
                            filterSubTree);
    visitor(op);
    return visitor.result();
}

} // namespace freetensor
