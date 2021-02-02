#include <algorithm>
#include <climits>
#include <functional>

#include <analyze/bounds.h>
#include <pass/disambiguous.h>

namespace ir {

Bound::Bound(const Expr &expr)
    : expr_(expr), lin_{{{getHash(expr), {1, expr}}}, 0} {}

Bound::Bound(const LinearExpr &lin) : lin_(lin) {
    Expr b = makeIntConst(lin.bias_);
    for (auto &&item : lin.coeff_) {
        int k = item.second.k;
        auto &&a = item.second.a;

        if (k == 0) {
            continue;
        }
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

        b = std::move(x);
    }
    expr_ = std::move(b);
}

std::vector<Bound> AnalyzeBounds::getLower(const Expr &op) const {
    if (lower_.count(op)) {
        return lower_.at(op);
    } else {
        return {};
    }
}

std::vector<Bound> AnalyzeBounds::getUpper(const Expr &op) const {
    if (upper_.count(op)) {
        return upper_.at(op);
    } else {
        return {};
    }
}

void AnalyzeBounds::updLower(const Expr &op, const Bound &bound) {
    if (!lower_.count(op)) {
        lower_[op] = {bound};
        return;
    }
    auto h = getHash(bound.expr_);
    for (Bound &old : lower_.at(op)) {
        if (getHash(old.expr_) == h) {
            return;
        }
        if (bound.expr_->nodeType() == ASTNodeType::IntConst &&
            old.expr_->nodeType() == ASTNodeType::IntConst) {
            auto oldVal = old.expr_.as<IntConstNode>()->val_;
            auto newVal = bound.expr_.as<IntConstNode>()->val_;
            if (newVal > oldVal) {
                old = Bound(LinearExpr{{}, newVal});
            }
            return;
        }
    }
    lower_.at(op).emplace_back(bound);
}

void AnalyzeBounds::updUpper(const Expr &op, const Bound &bound) {
    if (!upper_.count(op)) {
        upper_[op] = {bound};
        return;
    }
    auto h = getHash(bound.expr_);
    for (Bound &old : upper_.at(op)) {
        if (getHash(old.expr_) == h) {
            return;
        }
        if (bound.expr_->nodeType() == ASTNodeType::IntConst &&
            old.expr_->nodeType() == ASTNodeType::IntConst) {
            auto oldVal = old.expr_.as<IntConstNode>()->val_;
            auto newVal = bound.expr_.as<IntConstNode>()->val_;
            if (newVal < oldVal) {
                old = Bound(LinearExpr{{}, newVal});
            }
            return;
        }
    }
    upper_.at(op).emplace_back(bound);
}

int AnalyzeBounds::getIntLower(const Expr &op) const {
    int ret = INT_MIN;
    for (auto &&b : getLower(op)) {
        if (b.expr_->nodeType() == ASTNodeType::IntConst) {
            ret = std::max(ret, b.expr_.as<IntConstNode>()->val_);
        }
    }
    return ret;
}

int AnalyzeBounds::getIntUpper(const Expr &op) const {
    int ret = INT_MAX;
    for (auto &&b : getUpper(op)) {
        if (b.expr_->nodeType() == ASTNodeType::IntConst) {
            ret = std::min(ret, b.expr_.as<IntConstNode>()->val_);
        }
    }
    return ret;
}

Ref<int> AnalyzeBounds::getInt(const Expr &op) const {
    int lower = getIntLower(op);
    int upper = getIntUpper(op);
    return lower == upper ? Ref<int>::make(lower) : nullptr;
}

uint64_t AnalyzeBounds::getHash(const Expr &op) {
    if (hash_.count(op)) {
        return hash_.at(op);
    } else { // lowers / uppers are new exprs
        return ::ir::getHash(op);
    }
}

Expr AnalyzeBounds::sub1(const Expr &op) {
    if (op->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(op.as<IntConstNode>()->val_ - 1);
    } else {
        return makeSub(op, makeIntConst(1));
    }
}

Expr AnalyzeBounds::add1(const Expr &op) {
    if (op->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(op.as<IntConstNode>()->val_ + 1);
    } else {
        return makeAdd(op, makeIntConst(1));
    }
}

void AnalyzeBounds::visit(const VarDef &op) {

    for (auto &&dim : op->buffer_->tensor().shape()) {
        (*this)(dim);
    }
    if (buffers_.count(op->name_)) {
        throw InvalidProgram(
            "Conflict var name: " + op->name_ +
            ". Nested vars with the same name are not allowed");
    }
    buffers_[op->name_] = op->buffer_;
    (*this)(op->body_);
    buffers_.erase(op->name_);
}

void AnalyzeBounds::visit(const Var &op) {
    Visitor::visit(op);
    Bound b{op}; // Don't forget itself
    updLower(op, b);
    updUpper(op, b);
    static bool inRecur = false;
    if (!inRecur) {
        inRecur = true;
        if (iters_.count(op->name_)) {
            auto &&range = iters_.at(op->name_);
            auto first = disambiguous(range.first);
            auto second = disambiguous(range.second);
            (*this)(first);
            (*this)(second);
            for (auto &&item : getLower(first)) {
                updLower(op, item);
            }
            for (auto &&item : getUpper(second)) {
                updUpper(op, item);
            }
        }
        inRecur = false;
    }
}

void AnalyzeBounds::visit(const Load &op) {
    Visitor::visit(op);
    Bound b{op}; // Don't forget itself
    updLower(op, b);
    updUpper(op, b);
}

void AnalyzeBounds::visit(const IntConst &op) {
    Visitor::visit(op);
    Bound b{LinearExpr{{}, op->val_}};
    updLower(op, b);
    updUpper(op, b);
}

void AnalyzeBounds::visit(const Add &op) {
    Visitor::visit(op);
    auto f = [](const Bound &b1, const Bound &b2) -> Bound {
        auto ret = b1.lin_;
        for (auto &&item : b2.lin_.coeff_) {
            if (ret.coeff_.count(item.first)) {
                ret.coeff_[item.first].k += item.second.k;
            } else {
                ret.coeff_[item.first] = item.second;
            }
        }
        ret.bias_ += b2.lin_.bias_;
        return ret;
    };
    for (auto &&b1 : getLower(op->lhs_)) {
        for (auto &&b2 : getLower(op->rhs_)) {
            updLower(op, f(b1, b2));
        }
    }
    for (auto &&b1 : getUpper(op->lhs_)) {
        for (auto &&b2 : getUpper(op->rhs_)) {
            updUpper(op, f(b1, b2));
        }
    }
}

void AnalyzeBounds::visit(const Sub &op) {
    Visitor::visit(op);
    auto f = [](const Bound &b1, const Bound &b2) -> Bound {
        auto ret = b1.lin_;
        for (auto &&item : b2.lin_.coeff_) {
            if (ret.coeff_.count(item.first)) {
                ret.coeff_[item.first].k -= item.second.k;
            } else {
                ret.coeff_[item.first] = {-item.second.k, item.second.a};
            }
        }
        ret.bias_ -= b2.lin_.bias_;
        return ret;
    };
    for (auto &&b1 : getLower(op->lhs_)) {
        for (auto &&b2 : getUpper(op->rhs_)) {
            updLower(op, f(b1, b2));
        }
    }
    for (auto &&b1 : getUpper(op->lhs_)) {
        for (auto &&b2 : getLower(op->rhs_)) {
            updUpper(op, f(b1, b2));
        }
    }
}

void AnalyzeBounds::visit(const Mul &op) {
    Visitor::visit(op);

    // we deal with multiplying constant only. Otherwise, the extreme value of
    // `x * y` may not falls in the extreme value of `x` and `y`
    auto f = [](const Bound &b, int k) -> Bound {
        auto ret = b.lin_;
        for (auto &&item : ret.coeff_) {
            item.second.k *= k;
        }
        ret.bias_ *= k;
        return ret;
    };

    // FIXME: What if b < 0?
    auto g = [f, this](const Expr &op, const Expr &e1, const Expr &e2) {
        if (auto k = getInt(e2); k.isValid()) {
            for (auto &&b : getLower(e1)) {
                auto upd = *k > 0 ? &AnalyzeBounds::updLower
                                  : &AnalyzeBounds::updUpper;
                (this->*upd)(op, f(b, *k));
            }
            for (auto &&b : getUpper(e1)) {
                auto upd = *k > 0 ? &AnalyzeBounds::updUpper
                                  : &AnalyzeBounds::updLower;
                (this->*upd)(op, f(b, *k));
            }
        }
    };
    g(op, op->lhs_, op->rhs_);
    g(op, op->rhs_, op->lhs_);

    // Special for `(n // p) * k`
    if (lower_.count(op)) {
        for (Bound &b : lower_.at(op)) {
            bool altered = false;
            LinearExpr lin;
            lin.bias_ = b.lin_.bias_;
            for (auto &&item : b.lin_.coeff_) {
                if (item.second.a->nodeType() == ASTNodeType::Div) {
                    auto div = item.second.a.as<DivNode>();
                    if (div->rhs_->nodeType() == ASTNodeType::IntConst) {
                        if (int p = div->rhs_.as<IntConstNode>()->val_;
                            item.second.k % p == 0) {
                            auto h = getHash(div->lhs_);
                            if (lin.coeff_.count(h)) {
                                lin.coeff_.at(h).k += item.second.k / p;
                            } else {
                                lin.coeff_[h] = {item.second.k / p, div->lhs_};
                            }
                            lin.bias_ -= (p - 1) * (item.second.k / p);
                            altered = true;
                            continue;
                        }
                    }
                }
                lin.coeff_[item.first] = item.second;
            }
            if (altered) {
                b = Bound(lin);
            }
        }
    }
    if (upper_.count(op)) {
        for (Bound &b : upper_.at(op)) {
            bool altered = false;
            LinearExpr lin;
            lin.bias_ = b.lin_.bias_;
            for (auto &&item : b.lin_.coeff_) {
                if (item.second.a->nodeType() == ASTNodeType::Div) {
                    auto div = item.second.a.as<DivNode>();
                    if (div->rhs_->nodeType() == ASTNodeType::IntConst) {
                        if (int p = div->rhs_.as<IntConstNode>()->val_;
                            item.second.k % p == 0) {
                            auto h = getHash(div->lhs_);
                            if (lin.coeff_.count(h)) {
                                lin.coeff_.at(h).k += item.second.k / p;
                            } else {
                                lin.coeff_[h] = {item.second.k / p, div->lhs_};
                            }
                            altered = true;
                            continue;
                        }
                    }
                }
                lin.coeff_[item.first] = item.second;
            }
            if (altered) {
                b = Bound(lin);
            }
        }
    }
}

void AnalyzeBounds::visit(const Div &op) {
    Visitor::visit(op);

    // we deal with dividing by constant only. Otherwise, the extreme value of
    // `x / y` may not falls in the extreme value of `x` and `y`
    auto f = [](const Bound &b, int k) -> Bound {
        auto ret = b.lin_;
        for (auto &&item : ret.coeff_) {
            if (item.second.k % k != 0) {
                goto fail;
            }
            item.second.k /= k;
        }
        if (ret.bias_ % k != 0) {
            goto fail;
        }
        ret.bias_ /= k;
        return ret;
    fail:
        return makeDiv(b.expr_, makeIntConst(k));
    };

    if (auto k = getInt(op->rhs_); k.isValid()) {
        for (auto &&b : getLower(op->lhs_)) {
            (this->*(*k > 0 ? &AnalyzeBounds::updLower
                            : &AnalyzeBounds::updUpper))(op, f(b, *k));
        }
        for (auto &&b : getUpper(op->lhs_)) {
            (this->*(*k > 0 ? &AnalyzeBounds::updUpper
                            : &AnalyzeBounds::updLower))(op, f(b, *k));
        }
    }
}

void AnalyzeBounds::visit(const For &op) {
    if (iters_.count(op->iter_)) {
        throw InvalidProgram(
            "iterators with the same name in nested loops are not allowed");
    }
    iters_[op->iter_] = {op->begin_, sub1(op->end_)};
    Visitor::visit(op);
    iters_.erase(op->iter_);
}

void AnalyzeBounds::visit(const If &op) {
    (*this)(op->cond_);

    auto oldMap = iters_;
    switch (op->cond_->nodeType()) {
    case ASTNodeType::LT: {
        auto lt = op->cond_.as<LTNode>();
        if (lt->lhs_->nodeType() == ASTNodeType::Var) {
            iters_[lt->lhs_.as<VarNode>()->name_].second = sub1(lt->rhs_);
        }
        if (lt->rhs_->nodeType() == ASTNodeType::Var) {
            iters_[lt->rhs_.as<VarNode>()->name_].first = add1(lt->lhs_);
        }
        break;
    }
    case ASTNodeType::GT: {
        auto gt = op->cond_.as<GTNode>();
        if (gt->lhs_->nodeType() == ASTNodeType::Var) {
            iters_[gt->lhs_.as<VarNode>()->name_].first = add1(gt->rhs_);
        }
        if (gt->rhs_->nodeType() == ASTNodeType::Var) {
            iters_[gt->rhs_.as<VarNode>()->name_].second = sub1(gt->lhs_);
        }
        break;
    }
    case ASTNodeType::LE: {
        auto le = op->cond_.as<LENode>();
        if (le->lhs_->nodeType() == ASTNodeType::Var) {
            iters_[le->lhs_.as<VarNode>()->name_].second = le->rhs_;
        }
        if (le->rhs_->nodeType() == ASTNodeType::Var) {
            iters_[le->rhs_.as<VarNode>()->name_].first = le->lhs_;
        }
        break;
    }
    case ASTNodeType::GE: {
        auto ge = op->cond_.as<GENode>();
        if (ge->lhs_->nodeType() == ASTNodeType::Var) {
            iters_[ge->lhs_.as<VarNode>()->name_].first = ge->rhs_;
        }
        if (ge->rhs_->nodeType() == ASTNodeType::Var) {
            iters_[ge->rhs_.as<VarNode>()->name_].second = ge->lhs_;
        }
        break;
    }
    case ASTNodeType::EQ: {
        auto eq = op->cond_.as<EQNode>();
        if (eq->lhs_->nodeType() == ASTNodeType::Var) {
            iters_[eq->lhs_.as<VarNode>()->name_] = {eq->rhs_, eq->rhs_};
        }
        if (eq->rhs_->nodeType() == ASTNodeType::Var) {
            iters_[eq->rhs_.as<VarNode>()->name_] = {eq->lhs_, eq->lhs_};
        }
        break;
    }
    default:;
        // Do nothing
    }
    (*this)(op->thenCase_);
    iters_ = oldMap;

    if (op->elseCase_.isValid()) {
        auto oldMap = iters_;
        switch (op->cond_->nodeType()) {
        case ASTNodeType::GE: { // not LT
            auto lt = op->cond_.as<GENode>();
            if (lt->lhs_->nodeType() == ASTNodeType::Var) {
                iters_[lt->lhs_.as<VarNode>()->name_].second = sub1(lt->rhs_);
            }
            if (lt->rhs_->nodeType() == ASTNodeType::Var) {
                iters_[lt->rhs_.as<VarNode>()->name_].first = add1(lt->lhs_);
            }
            break;
        }
        case ASTNodeType::LE: { // not GT
            auto gt = op->cond_.as<LENode>();
            if (gt->lhs_->nodeType() == ASTNodeType::Var) {
                iters_[gt->lhs_.as<VarNode>()->name_].first = add1(gt->rhs_);
            }
            if (gt->rhs_->nodeType() == ASTNodeType::Var) {
                iters_[gt->rhs_.as<VarNode>()->name_].second = sub1(gt->lhs_);
            }
            break;
        }
        case ASTNodeType::GT: { // not LE
            auto le = op->cond_.as<GTNode>();
            if (le->lhs_->nodeType() == ASTNodeType::Var) {
                iters_[le->lhs_.as<VarNode>()->name_].second = le->rhs_;
            }
            if (le->rhs_->nodeType() == ASTNodeType::Var) {
                iters_[le->rhs_.as<VarNode>()->name_].first = le->lhs_;
            }
            break;
        }
        case ASTNodeType::LT: { // not GE
            auto ge = op->cond_.as<LTNode>();
            if (ge->lhs_->nodeType() == ASTNodeType::Var) {
                iters_[ge->lhs_.as<VarNode>()->name_].first = ge->rhs_;
            }
            if (ge->rhs_->nodeType() == ASTNodeType::Var) {
                iters_[ge->rhs_.as<VarNode>()->name_].second = ge->lhs_;
            }
            break;
        }
        case ASTNodeType::NE: { // not EQ
            auto eq = op->cond_.as<NENode>();
            if (eq->lhs_->nodeType() == ASTNodeType::Var) {
                iters_[eq->lhs_.as<VarNode>()->name_] = {eq->rhs_, eq->rhs_};
            }
            if (eq->rhs_->nodeType() == ASTNodeType::Var) {
                iters_[eq->rhs_.as<VarNode>()->name_] = {eq->lhs_, eq->lhs_};
            }
            break;
        }
        default:;
            // Do nothing
        }
        (*this)(op->elseCase_);
        iters_ = oldMap;
    }
}

} // namespace ir
