#include <analyze/linear.h>

namespace ir {

void AnalyzeLinear::visit(const Var &op) {
    Visitor::visit(op);
    result_[op] = {{{hash_.at(op), {1, op}}}, 0};
}

void AnalyzeLinear::visit(const Load &op) {
    Visitor::visit(op);
    result_[op] = {{{hash_.at(op), {1, op}}}, 0};
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

    auto ret = e1;
    for (auto &&item : e2.coeff_) {
        if (ret.coeff_.count(item.first)) {
            ret.coeff_[item.first].k += item.second.k;
        } else {
            ret.coeff_[item.first] = item.second;
        }
    }
    ret.bias_ += e2.bias_;
    result_[op] = ret;
}

void AnalyzeLinear::visit(const Sub &op) {
    Visitor::visit(op);
    if (!result_.count(op->lhs_) || !result_.count(op->rhs_)) {
        return;
    }
    const auto &e1 = result_.at(op->lhs_);
    const auto &e2 = result_.at(op->rhs_);

    auto ret = e1;
    for (auto &&item : e2.coeff_) {
        if (ret.coeff_.count(item.first)) {
            ret.coeff_[item.first].k -= item.second.k;
        } else {
            ret.coeff_[item.first] = {-item.second.k, item.second.a};
        }
    }
    ret.bias_ -= e2.bias_;
    result_[op] = ret;
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
            item.second.k *= e1.bias_;
        }
        ret.bias_ *= e1.bias_;
        result_[op] = ret;
        return;
    }
    if (e2.coeff_.empty()) {
        auto ret = e1;
        for (auto &&item : ret.coeff_) {
            item.second.k *= e2.bias_;
        }
        ret.bias_ *= e2.bias_;
        result_[op] = ret;
        return;
    }
    // Not linear
}

Ref<LinearExpr> analyzeLinear(const Expr &op) {
    auto hash = getHashMap(op);
    AnalyzeLinear analyzeLinear(hash);
    analyzeLinear(op);
    auto &&linear = analyzeLinear.result();
    return linear.count(op) ? Ref<LinearExpr>::make(linear.at(op)) : nullptr;
}

} // namespace ir

