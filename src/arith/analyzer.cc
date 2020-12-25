#include <functional>

#include <arith/analyzer.h>

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
            item.second.k /= e2.bias_;
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
    } else if (k == 1) {
        x = a;
    } else {
        x = makeMul(makeIntConst(k), a);
    }
    if (x->nodeType() == ASTNodeType::IntConst &&
        b->nodeType() == ASTNodeType::IntConst) {
        x = makeIntConst(x.as<IntConstNode>()->val_ +
                         b.as<IntConstNode>()->val_);
    } else if (b->nodeType() == ASTNodeType::IntConst &&
               b.as<IntConstNode>()->val_ == 0) {
        // do nothing
    } else {
        x = makeAdd(x, b);
    }
    return x;
}

std::vector<Expr> AnalyzeBounds::getLower(const LinearExpr &linear) const {
    std::vector<Expr> ret;
    typedef std::unordered_map<uint64_t, Scale>::const_iterator Iter;
    std::function<void(Iter, Expr)> dfs = [&](Iter i, Expr expr) {
        if (i == linear.coeff_.end()) {
            ret.emplace_back(expr);
            return;
        }
        auto ii = i;
        ii++;
        if (i->second.k > 0 && lower_.count(i->second.a.get())) {
            for (auto &&candidate : lower_.at(i->second.a.get())) {
                dfs(ii, compLinear(i->second.k, candidate, expr));
            }
        }
        if (i->second.k < 0 && upper_.count(i->second.a.get())) {
            for (auto &&candidate : upper_.at(i->second.a.get())) {
                dfs(ii, compLinear(i->second.k, candidate, expr));
            }
        }
        if (i->second.k == 0) {
            dfs(ii, expr);
        }
    };
    dfs(linear.coeff_.begin(), makeIntConst(linear.bias_));
    return ret;
}

std::vector<Expr> AnalyzeBounds::getUpper(const LinearExpr &linear) const {
    std::vector<Expr> ret;
    typedef std::unordered_map<uint64_t, Scale>::const_iterator Iter;
    std::function<void(Iter, Expr)> dfs = [&](Iter i, Expr expr) {
        if (i == linear.coeff_.end()) {
            ret.emplace_back(expr);
            return;
        }
        auto ii = i;
        ii++;
        if (i->second.k > 0 && upper_.count(i->second.a.get())) {
            for (auto &&candidate : upper_.at(i->second.a.get())) {
                dfs(ii, compLinear(i->second.k, candidate, expr));
            }
        }
        if (i->second.k < 0 && lower_.count(i->second.a.get())) {
            for (auto &&candidate : lower_.at(i->second.a.get())) {
                dfs(ii, compLinear(i->second.k, candidate, expr));
            }
        }
        if (i->second.k == 0) {
            dfs(ii, expr);
        }
    };
    dfs(linear.coeff_.begin(), makeIntConst(linear.bias_));
    return ret;
}

void AnalyzeBounds::updLower(const Expr &op, const std::vector<Expr> &exprs) {
    if (!lower_.count(op.get())) {
        lower_[op.get()] = exprs;
        return;
    }
    for (auto &&expr : exprs) {
        auto h = getHash(expr);
        for (Expr &old : lower_.at(op.get())) {
            if (getHash(old) == h) {
                goto done;
            }
            if (expr->nodeType() == ASTNodeType::IntConst &&
                old->nodeType() == ASTNodeType::IntConst) {
                auto oldVal = old.as<IntConstNode>()->val_;
                auto newVal = expr.as<IntConstNode>()->val_;
                if (newVal > oldVal) {
                    old.as<IntConstNode>()->val_ = newVal;
                }
                goto done;
            }
        }
        lower_.at(op.get()).emplace_back(expr);
    done:;
    }
}

void AnalyzeBounds::updUpper(const Expr &op, const std::vector<Expr> &exprs) {
    if (!upper_.count(op.get())) {
        upper_[op.get()] = exprs;
        return;
    }
    for (auto &&expr : exprs) {
        auto h = getHash(expr);
        for (Expr &old : upper_.at(op.get())) {
            if (getHash(old) == h) {
                goto done;
            }
            if (expr->nodeType() == ASTNodeType::IntConst &&
                old->nodeType() == ASTNodeType::IntConst) {
                auto oldVal = old.as<IntConstNode>()->val_;
                auto newVal = expr.as<IntConstNode>()->val_;
                if (newVal < oldVal) {
                    old.as<IntConstNode>()->val_ = newVal;
                }
                goto done;
            }
        }
        upper_.at(op.get()).emplace_back(expr);
    done:;
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
    if (buffers_.count(op->name_)) {
        ERROR("Conflict var name: " + op->name_ +
              ". Nested vars with the same name are not allowed");
    }
    buffers_[op->name_] = op->buffer_;
    (*this)(op->body_);
    buffers_.erase(op->name_);
}

void AnalyzeBounds::visit(const Var &op) {
    Visitor::visit(op);
    updLower(op, {op}); // Don't forget itself
    updUpper(op, {op});
    if (iters_.count(op->name_)) {
        updLower(op, {iters_[op->name_].first});
        updUpper(op, {iters_[op->name_].second});
    }
}

void AnalyzeBounds::visit(const Store &op) {
    Visitor::visit(op); // Recurse first, assume there won't be the same buffer
                        // in the indices
    if (!buffers_.count(op->var_)) {
        ERROR("Storing to undefined variable " + op->var_);
    }
    for (size_t i = 0, iEnd = op->indices_.size(); i < iEnd; i++) {
        auto &&shape = buffers_.at(op->var_)->tensor().shape();
        updLower(op->indices_[i], {makeIntConst(0)});
        updUpper(op->indices_[i], {shape[i]});
    }
}

void AnalyzeBounds::visit(const Load &op) {
    Visitor::visit(op);
    updLower(op, {op}); // Don't forget itself
    updUpper(op, {op});
    if (!buffers_.count(op->var_)) {
        ERROR("Storing to undefined variable " + op->var_);
    }
    for (size_t i = 0, iEnd = op->indices_.size(); i < iEnd; i++) {
        auto &&shape = buffers_.at(op->var_)->tensor().shape();
        updLower(op->indices_[i], {makeIntConst(0)});
        updUpper(op->indices_[i], {shape[i]});
    }
}

void AnalyzeBounds::visit(const IntConst &op) { doAnalyze(op); }
void AnalyzeBounds::visit(const Add &op) { doAnalyze(op); }
void AnalyzeBounds::visit(const Sub &op) { doAnalyze(op); }
void AnalyzeBounds::visit(const Mul &op) { doAnalyze(op); }
void AnalyzeBounds::visit(const Div &op) { doAnalyze(op); }

void AnalyzeBounds::visit(const For &op) {
    if (iters_.count(op->iter_)) {
        ERROR("iterators with the same name in nested loops are not allowed");
    }
    iters_[op->iter_].first = op->begin_;
    if (op->end_->nodeType() == ASTNodeType::IntConst) {
        iters_[op->iter_].second =
            makeIntConst(op->end_.as<IntConstNode>()->val_ - 1);
    } else {
        iters_[op->iter_].second = makeSub(op->end_, makeIntConst(1));
    }
    Visitor::visit(op);
    iters_.erase(op->iter_);
}

} // namespace ir

