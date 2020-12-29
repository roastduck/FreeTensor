#include <analyze/linear.h>

namespace ir {

void AnalyzeLinear::visit(const Var &op) {
    Visitor::visit(op);
    result_[op.get()] = {{{hash_.at(op.get()), {1, op}}}, 0};
}

void AnalyzeLinear::visit(const Load &op) {
    Visitor::visit(op);
    result_[op.get()] = {{{hash_.at(op.get()), {1, op}}}, 0};
}

void AnalyzeLinear::visit(const IntConst &op) {
    Visitor::visit(op);
    result_[op.get()] = {{}, op->val_};
}

void AnalyzeLinear::visit(const Add &op) {
    Visitor::visit(op);
    if (!result_.count(op->lhs_.get()) || !result_.count(op->rhs_.get())) {
        return;
    }
    const auto &e1 = result_.at(op->lhs_.get());
    const auto &e2 = result_.at(op->rhs_.get());

    auto ret = e1;
    for (auto &&item : e2.coeff_) {
        if (ret.coeff_.count(item.first)) {
            ret.coeff_[item.first].k += item.second.k;
        } else {
            ret.coeff_[item.first] = item.second;
        }
    }
    ret.bias_ += e2.bias_;
    result_[op.get()] = ret;
}

void AnalyzeLinear::visit(const Sub &op) {
    Visitor::visit(op);
    if (!result_.count(op->lhs_.get()) || !result_.count(op->rhs_.get())) {
        return;
    }
    const auto &e1 = result_.at(op->lhs_.get());
    const auto &e2 = result_.at(op->rhs_.get());

    auto ret = e1;
    for (auto &&item : e2.coeff_) {
        if (ret.coeff_.count(item.first)) {
            ret.coeff_[item.first].k -= item.second.k;
        } else {
            ret.coeff_[item.first] = {-item.second.k, item.second.a};
        }
    }
    ret.bias_ -= e2.bias_;
    result_[op.get()] = ret;
}

void AnalyzeLinear::visit(const Mul &op) {
    Visitor::visit(op);
    if (!result_.count(op->lhs_.get()) || !result_.count(op->rhs_.get())) {
        return;
    }
    const auto &e1 = result_.at(op->lhs_.get());
    const auto &e2 = result_.at(op->rhs_.get());

    if (e1.coeff_.empty()) {
        auto ret = e2;
        for (auto &&item : ret.coeff_) {
            item.second.k *= e1.bias_;
        }
        ret.bias_ *= e1.bias_;
        result_[op.get()] = ret;
        return;
    }
    if (e2.coeff_.empty()) {
        auto ret = e1;
        for (auto &&item : ret.coeff_) {
            item.second.k *= e2.bias_;
        }
        ret.bias_ *= e2.bias_;
        result_[op.get()] = ret;
        return;
    }
    // Not linear
}

} // namespace ir

