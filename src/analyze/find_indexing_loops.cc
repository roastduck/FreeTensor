#include <analyze/find_indexing_loops.h>

namespace freetensor {

void FindIndexingLoops::visit(const Load &op) {
    inIndicesStack_.emplace_back(def(op->var_));
    for (auto &&idx : op->indices_) {
        (*this)(idx);
    }
    inIndicesStack_.pop_back();
}

void FindIndexingLoops::visit(const Store &op) {
    inIndicesStack_.emplace_back(def(op->var_));
    for (auto &&idx : op->indices_) {
        (*this)(idx);
    }
    inIndicesStack_.pop_back();
    (*this)(op->expr_);
}

void FindIndexingLoops::visit(const ReduceTo &op) {
    inIndicesStack_.emplace_back(def(op->var_));
    for (auto &&idx : op->indices_) {
        (*this)(idx);
    }
    inIndicesStack_.pop_back();
    (*this)(op->expr_);
}

void FindIndexingLoops::visit(const Var &op) {
    Visitor::visit(op);
    for (auto &&def : inIndicesStack_) {
        results_[loop(op->name_)].emplace_back(def);
    }
}

} // namespace freetensor
