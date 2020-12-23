#include <arith/analyzer.h>

namespace ir {

void AnalyzeLinear::visit(const Var &op) {
    Visitor::visit(op);
    result_[op.get()] = {{{hash_.at(op.get()), 1}}, 0};
}

void AnalyzeLinear::visit(const Load &op) {
    Visitor::visit(op);
    result_[op.get()] = {{{hash_.at(op.get()), 1}}, 0};
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
            ret.coeff_[item.first] += item.second;
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
            ret.coeff_[item.first] -= item.second;
        } else {
            ret.coeff_[item.first] = -item.second;
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
            item.second *= e1.bias_;
        }
        ret.bias_ *= e1.bias_;
        result_[op.get()] = ret;
        return;
    }
    if (e2.coeff_.empty()) {
        auto ret = e1;
        for (auto &&item : ret.coeff_) {
            item.second *= e2.bias_;
        }
        ret.bias_ *= e2.bias_;
        result_[op.get()] = ret;
        return;
    }
    // Not linear
}

void AnalyzeLinear::visit(const Div &op) {
    Visitor::visit(op);
    if (!result_.count(op->lhs_.get()) || !result_.count(op->rhs_.get())) {
        return;
    }
    const auto &e1 = result_.at(op->lhs_.get());
    const auto &e2 = result_.at(op->rhs_.get());

    if (e2.coeff_.empty()) {
        auto ret = e1;
        for (auto &&item : ret.coeff_) {
            item.second /= e2.bias_;
        }
        ret.bias_ /= e2.bias_;
        result_[op.get()] = ret;
        return;
    }
    // Not linear
}

Expr AnalyzeBounds::compLinear(int k, const Expr &a, const Expr &b) const {
    Expr x;
    if (a->nodeType() == ASTNodeType::IntConst) {
        x = makeIntConst(k * a.as<IntConstNode>()->val_);
    } else {
        x = makeMul(makeIntConst(k), a);
    }
    if (x->nodeType() == ASTNodeType::IntConst &&
        b->nodeType() == ASTNodeType::IntConst) {
        x = makeIntConst(x.as<IntConstNode>()->val_ +
                         b.as<IntConstNode>()->val_);
    } else {
        x = makeAdd(x, b);
    }
    return x;
}

Expr AnalyzeBounds::getLower(const LinearExpr &linear) const {
    Expr ret = makeIntConst(linear.bias_);
    for (auto &&item : linear.coeff_) {
        if (item.second > 0) {
            if (!lower_.count(item.first)) {
                return nullptr;
            }
            ret = compLinear(item.second, lower_.at(item.first), ret);
        }
        if (item.second < 0) {
            if (!upper_.count(item.first)) {
                return nullptr;
            }
            ret = compLinear(item.second, upper_.at(item.first), ret);
        }
    }
    return ret;
}

Expr AnalyzeBounds::getUpper(const LinearExpr &linear) const {
    Expr ret = makeIntConst(linear.bias_);
    for (auto &&item : linear.coeff_) {
        if (item.second > 0) {
            if (!upper_.count(item.first)) {
                return nullptr;
            }
            ret = compLinear(item.second, upper_.at(item.first), ret);
        }
        if (item.second < 0) {
            if (!lower_.count(item.first)) {
                return nullptr;
            }
            ret = compLinear(item.second, lower_.at(item.first), ret);
        }
    }
    return ret;
}

void AnalyzeBounds::doAnalyze(const Expr &op) {
    if (linear_.count(op.get())) {
        auto &&lin = linear_.at(op.get());
        auto lower = getLower(lin), upper = getUpper(lin);
        auto h = hash_.at(op.get());
        if (lower.isValid()) {
            lower_[h] = lower;
        }
        if (upper.isValid()) {
            upper_[h] = upper;
        }
    }
}

void AnalyzeBounds::updLower(uint64_t hash, const Expr &expr) {
    if (!lower_.count(hash)) {
        lower_[hash] = expr;
    } else if (getHash(lower_.at(hash)) == getHash(expr)) {
        return;
    } else if (expr->nodeType() == ASTNodeType::IntConst &&
               lower_.at(hash)->nodeType() == ASTNodeType::IntConst) {
        auto oldVal = lower_.at(hash).as<IntConstNode>()->val_;
        auto newVal = expr.as<IntConstNode>()->val_;
        if (newVal > oldVal) {
            lower_[hash] = expr;
        }
    } else {
        // TODO: Use max node
        ASSERT(false);
    }
}

void AnalyzeBounds::updUpper(uint64_t hash, const Expr &expr) {
    if (!upper_.count(hash)) {
        upper_[hash] = expr;
    } else if (getHash(upper_.at(hash)) == getHash(expr)) {
        return;
    } else if (expr->nodeType() == ASTNodeType::IntConst &&
               upper_.at(hash)->nodeType() == ASTNodeType::IntConst) {
        auto oldVal = upper_.at(hash).as<IntConstNode>()->val_;
        auto newVal = expr.as<IntConstNode>()->val_;
        if (newVal < oldVal) {
            upper_[hash] = expr;
        }
    } else {
        // TODO: Use min node
        ASSERT(false);
    }
}

uint64_t AnalyzeBounds::getHash(const Expr &op) {
    if (hash_.count(op.get())) {
        return hash_.at(op.get());
    } else { // lowers / uppers are new exprs
        return ::ir::getHash(op);
    }
}

void AnalyzeBounds::visit(const VarDef &op) {
    for (auto &&dim : op->buffer_->tensor().shape()) {
        (*this)(dim);
    }
    if (vars_.count(op->name_)) {
        ERROR("Conflict var name: " + op->name_ +
              ". Vars with the same name, even not nested, is not allowed");
    }
    vars_[op->name_] = op->buffer_;
    (*this)(op->body_);
}

void AnalyzeBounds::visit(const Var &op) { doAnalyze(op); }

void AnalyzeBounds::visit(const Store &op) {
    if (!vars_.count(op->var_)) {
        ERROR("Storing to undefined variable " + op->var_);
    }
    for (size_t i = 0, iEnd = op->indices_.size(); i < iEnd; i++) {
        auto &&shape = vars_.at(op->var_)->tensor().shape();
        updLower(hash_.at(op->indices_[i].get()), makeIntConst(0));
        updUpper(hash_.at(op->indices_[i].get()), shape[i]);
    }
    Visitor::visit(op);
}

void AnalyzeBounds::visit(const Load &op) {
    if (!vars_.count(op->var_)) {
        ERROR("Storing to undefined variable " + op->var_);
    }
    for (size_t i = 0, iEnd = op->indices_.size(); i < iEnd; i++) {
        auto &&shape = vars_.at(op->var_)->tensor().shape();
        updLower(hash_.at(op->indices_[i].get()), makeIntConst(0));
        updUpper(hash_.at(op->indices_[i].get()), shape[i]);
    }
    Visitor::visit(op);
}

void AnalyzeBounds::visit(const IntConst &op) { doAnalyze(op); }
void AnalyzeBounds::visit(const Add &op) { doAnalyze(op); }
void AnalyzeBounds::visit(const Sub &op) { doAnalyze(op); }
void AnalyzeBounds::visit(const Mul &op) { doAnalyze(op); }
void AnalyzeBounds::visit(const Div &op) { doAnalyze(op); }

void AnalyzeBounds::visit(const For &op) {
    auto h = getHash(makeVar(op->iter_));
    updLower(h, op->begin_);
    updUpper(h, op->end_);
    Visitor::visit(op);
}

} // namespace ir

