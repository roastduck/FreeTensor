#include <analyze/analyze_linear.h>

namespace ir {

void AnalyzeLinear::visitExpr(
    const Expr &op, const std::function<void(const Expr &)> &visitNode) {
    if (!visited_.count(op)) {
        visited_.insert(op);
        Visitor::visitExpr(op, visitNode);
    }
}

void AnalyzeLinear::visit(const Var &op) {
    Visitor::visit(op);
    getHash_(op);
    result_[op] = {{{getHash_.hash().at(op), {1, op}}}, 0};
}

void AnalyzeLinear::visit(const Load &op) {
    Visitor::visit(op);
    getHash_(op);
    result_[op] = {{{getHash_.hash().at(op), {1, op}}}, 0};
}

void AnalyzeLinear::visit(const IntConst &op) {
    Visitor::visit(op);
    result_[op] = {{}, op->val_};
}

void AnalyzeLinear::visit(const Add &op) {
    Visitor::visit(op);
    if (!result_.count(op->lhs_) || !result_.count(op->rhs_)) {
        return;
    }
    const auto &e1 = result_.at(op->lhs_);
    const auto &e2 = result_.at(op->rhs_);
    result_[op] = add(e1, e2);
}

void AnalyzeLinear::visit(const Sub &op) {
    Visitor::visit(op);
    if (!result_.count(op->lhs_) || !result_.count(op->rhs_)) {
        return;
    }
    const auto &e1 = result_.at(op->lhs_);
    const auto &e2 = result_.at(op->rhs_);
    result_[op] = sub(e1, e2);
}

void AnalyzeLinear::visit(const Mul &op) {
    Visitor::visit(op);
    if (!result_.count(op->lhs_) || !result_.count(op->rhs_)) {
        return;
    }
    const auto &e1 = result_.at(op->lhs_);
    const auto &e2 = result_.at(op->rhs_);

    if (e1.coeff_.empty()) {
        auto ret = e2;
        for (auto &&item : ret.coeff_) {
            item.second.k_ *= e1.bias_;
        }
        ret.bias_ *= e1.bias_;
        result_[op] = ret;
        return;
    }
    if (e2.coeff_.empty()) {
        auto ret = e1;
        for (auto &&item : ret.coeff_) {
            item.second.k_ *= e2.bias_;
        }
        ret.bias_ *= e2.bias_;
        result_[op] = ret;
        return;
    }
    // Not linear
}

} // namespace ir

