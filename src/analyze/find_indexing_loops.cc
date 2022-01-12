#include <analyze/find_indexing_loops.h>

namespace ir {

void FindIndexingLoops::visit(const For &op) {
    ASSERT(!loops_.count(op->iter_));
    loops_[op->iter_] = op;
    Visitor::visit(op);
    loops_.erase(op->iter_);
}

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
        results_[loops_.at(op->name_)].emplace_back(def);
    }
}

} // namespace ir
