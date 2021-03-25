#include <analyze/find_loop_variance.h>

namespace ir {

void MarkStores::visit(const For &op) {
    loopStack_.emplace_back(op->id());
    Visitor::visit(op);
    loopStack_.pop_back();
}

void MarkStores::visit(const Store &op) {
    Visitor::visit(op);
    for (auto &&loop : loopStack_) {
        variantVar_[op->var_].insert(loop);
    }
}

void MarkStores::visit(const ReduceTo &op) {
    Visitor::visit(op);
    for (auto &&loop : loopStack_) {
        variantVar_[op->var_].insert(loop);
    }
}

void FindLoopVariance::visit(const For &op) {
    loopStack_.emplace_back(op->id());
    variantVar_[op->iter_].insert(op->id());

    Visitor::visit(op);

    variantVar_.erase(op->iter_);
    loopStack_.pop_back();
}

void FindLoopVariance::visit(const VarDef &op) {
    for (auto &&loop : loopStack_) {
        variantVar_[op->name_].insert(loop);
    }
    MarkStores{variantVar_}(op);

    Visitor::visit(op);

    variantVar_.erase(op->name_);
}

void FindLoopVariance::visit(const Var &op) {
    Visitor::visit(op);
    if (variantVar_.count(op->name_)) {
        variantExpr_[op] = variantVar_.at(op->name_);
    }
}

void FindLoopVariance::visit(const Load &op) {
    Visitor::visit(op);
    if (variantVar_.count(op->var_)) {
        variantExpr_[op] = variantVar_.at(op->var_);
    }
    for (auto &&index : op->indices_) {
        if (variantExpr_.count(index)) {
            for (auto &&loop : variantExpr_.at(index)) {
                variantExpr_[op].insert(loop);
            }
        }
    }
}

void FindLoopVariance::visit(const LNot &op) {
    Visitor::visit(op);
    if (variantExpr_.count(op->expr_)) {
        variantExpr_[op] = variantExpr_.at(op->expr_);
    }
}

std::unordered_map<Expr, std::unordered_set<std::string>>
findLoopVariance(const AST &op) {
    FindLoopVariance visitor;
    visitor(op);
    return visitor.variantExpr();
}

} // namespace ir

