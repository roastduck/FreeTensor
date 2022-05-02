#include <analyze/as_dnf.h>

namespace freetensor {

template <class T>
static std::vector<T> merge(const std::vector<T> &lhs,
                            const std::vector<T> &rhs) {
    std::vector<T> ret;
    ret.reserve(lhs.size() + rhs.size());
    ret.insert(ret.end(), lhs.begin(), lhs.end());
    ret.insert(ret.end(), rhs.begin(), rhs.end());
    return ret;
}

void AsDNF::visitExpr(const Expr &op) {
    Visitor::visitExpr(op);
    if (!results_.count(op)) {
        results_[op] = {{op}};
    }
}

void AsDNF::visit(const LAnd &op) {
    Visitor::visit(op);
    if (!neg_) {
        for (auto &&l : results_.at(op->lhs_)) {
            for (auto &&r : results_.at(op->rhs_)) {
                results_[op].emplace_back(merge(l, r));
            }
        }
    } else {
        results_[op] = merge(results_.at(op->lhs_), results_.at(op->rhs_));
    }
}

void AsDNF::visit(const LOr &op) {
    Visitor::visit(op);
    if (!neg_) {
        results_[op] = merge(results_.at(op->lhs_), results_.at(op->rhs_));
    } else {
        for (auto &&l : results_.at(op->lhs_)) {
            for (auto &&r : results_.at(op->rhs_)) {
                results_[op].emplace_back(merge(l, r));
            }
        }
    }
}

void AsDNF::visit(const LNot &op) {
    neg_ = !neg_;
    Visitor::visit(op);
    results_[op] = results_.at(op->expr_);
    neg_ = !neg_;
}

void AsDNF::visit(const EQ &op) {
    if (!neg_) {
        results_[op] = {{op}};
    } else {
        results_[op] = {{makeNE(op->lhs_, op->rhs_)}};
    }
}

void AsDNF::visit(const NE &op) {
    if (!neg_) {
        results_[op] = {{op}};
    } else {
        results_[op] = {{makeEQ(op->lhs_, op->rhs_)}};
    }
}

void AsDNF::visit(const LT &op) {
    if (!neg_) {
        results_[op] = {{op}};
    } else {
        results_[op] = {{makeGE(op->lhs_, op->rhs_)}};
    }
}

void AsDNF::visit(const LE &op) {
    if (!neg_) {
        results_[op] = {{op}};
    } else {
        results_[op] = {{makeGT(op->lhs_, op->rhs_)}};
    }
}

void AsDNF::visit(const GT &op) {
    if (!neg_) {
        results_[op] = {{op}};
    } else {
        results_[op] = {{makeLE(op->lhs_, op->rhs_)}};
    }
}

void AsDNF::visit(const GE &op) {
    if (!neg_) {
        results_[op] = {{op}};
    } else {
        results_[op] = {{makeLT(op->lhs_, op->rhs_)}};
    }
}

DNF asDNF(const Expr &expr) {
    AsDNF visitor;
    visitor(expr);
    return visitor.results(expr);
}

} // namespace freetensor
