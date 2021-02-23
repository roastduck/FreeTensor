#include <climits>
#include <sstream>
#include <unordered_set>

#include <analyze/hash.h>
#include <except.h>
#include <pass/disambiguous.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/simplify.h>

namespace ir {

static bool isEmptyStmt(const Stmt &op) {
    if (!op.isValid()) { // In case If->elseCase_ == nullptr
        return true;
    }
    if (op->nodeType() == ASTNodeType::StmtSeq &&
        op.as<StmtSeqNode>()->stmts_.empty()) {
        return true;
    }
    return false;
}

void FindInnerMostScope::visit(const Var &op) {
    Visitor::visit(op);
    innerMost_ = std::max(innerMost_, varScope_.at(op->name_));
}

void FindInnerMostScope::visit(const Load &op) {
    Visitor::visit(op);
    innerMost_ = std::max(innerMost_, varScope_.at(op->var_));
}

int findInnerMostScope(const std::unordered_map<std::string, int> &varScope,
                       const Expr &op) {
    FindInnerMostScope visitor(varScope);
    visitor(op);
    return visitor.innnerMost();
}

Expr CompIterBounds::sub1(const Expr &op) {
    if (op->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(op.as<IntConstNode>()->val_ - 1);
    } else {
        return makeSub(op, makeIntConst(1));
    }
}

Expr CompIterBounds::add1(const Expr &op) {
    if (op->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(op.as<IntConstNode>()->val_ + 1);
    } else {
        return makeAdd(op, makeIntConst(1));
    }
}

Stmt CompIterBounds::visit(const For &op) {
    if (iters_.count(op->iter_)) {
        throw InvalidProgram(
            "iterators with the same name in nested loops are not allowed");
    }
    iters_[op->iter_] = {op->begin_, sub1(op->end_)};
    auto ret = Mutator::visit(op);
    iters_.erase(op->iter_);
    return ret;
}

Stmt CompIterBounds::visit(const If &op) {
    auto cond = (*this)(op->cond_);
    auto notCond =
        op->infoNotCond_.isValid() ? (*this)(op->infoNotCond_) : nullptr;

    auto oldMap = iters_;
    switch (cond->nodeType()) {
    case ASTNodeType::LT: {
        auto lt = cond.as<LTNode>();
        if (lt->lhs_->nodeType() == ASTNodeType::Var) {
            iters_[lt->lhs_.as<VarNode>()->name_].second = sub1(lt->rhs_);
        }
        if (lt->rhs_->nodeType() == ASTNodeType::Var) {
            iters_[lt->rhs_.as<VarNode>()->name_].first = add1(lt->lhs_);
        }
        break;
    }
    case ASTNodeType::GT: {
        auto gt = cond.as<GTNode>();
        if (gt->lhs_->nodeType() == ASTNodeType::Var) {
            iters_[gt->lhs_.as<VarNode>()->name_].first = add1(gt->rhs_);
        }
        if (gt->rhs_->nodeType() == ASTNodeType::Var) {
            iters_[gt->rhs_.as<VarNode>()->name_].second = sub1(gt->lhs_);
        }
        break;
    }
    case ASTNodeType::LE: {
        auto le = cond.as<LENode>();
        if (le->lhs_->nodeType() == ASTNodeType::Var) {
            iters_[le->lhs_.as<VarNode>()->name_].second = le->rhs_;
        }
        if (le->rhs_->nodeType() == ASTNodeType::Var) {
            iters_[le->rhs_.as<VarNode>()->name_].first = le->lhs_;
        }
        break;
    }
    case ASTNodeType::GE: {
        auto ge = cond.as<GENode>();
        if (ge->lhs_->nodeType() == ASTNodeType::Var) {
            iters_[ge->lhs_.as<VarNode>()->name_].first = ge->rhs_;
        }
        if (ge->rhs_->nodeType() == ASTNodeType::Var) {
            iters_[ge->rhs_.as<VarNode>()->name_].second = ge->lhs_;
        }
        break;
    }
    case ASTNodeType::EQ: {
        auto eq = cond.as<EQNode>();
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
    auto thenCase = (*this)(op->thenCase_);
    iters_ = oldMap;

    Stmt elseCase = nullptr;
    if (op->elseCase_.isValid()) {
        auto oldMap = iters_;
        switch (cond->nodeType()) {
        case ASTNodeType::GE: { // not LT
            auto lt = cond.as<GENode>();
            if (lt->lhs_->nodeType() == ASTNodeType::Var) {
                iters_[lt->lhs_.as<VarNode>()->name_].second = sub1(lt->rhs_);
            }
            if (lt->rhs_->nodeType() == ASTNodeType::Var) {
                iters_[lt->rhs_.as<VarNode>()->name_].first = add1(lt->lhs_);
            }
            break;
        }
        case ASTNodeType::LE: { // not GT
            auto gt = cond.as<LENode>();
            if (gt->lhs_->nodeType() == ASTNodeType::Var) {
                iters_[gt->lhs_.as<VarNode>()->name_].first = add1(gt->rhs_);
            }
            if (gt->rhs_->nodeType() == ASTNodeType::Var) {
                iters_[gt->rhs_.as<VarNode>()->name_].second = sub1(gt->lhs_);
            }
            break;
        }
        case ASTNodeType::GT: { // not LE
            auto le = cond.as<GTNode>();
            if (le->lhs_->nodeType() == ASTNodeType::Var) {
                iters_[le->lhs_.as<VarNode>()->name_].second = le->rhs_;
            }
            if (le->rhs_->nodeType() == ASTNodeType::Var) {
                iters_[le->rhs_.as<VarNode>()->name_].first = le->lhs_;
            }
            break;
        }
        case ASTNodeType::LT: { // not GE
            auto ge = cond.as<LTNode>();
            if (ge->lhs_->nodeType() == ASTNodeType::Var) {
                iters_[ge->lhs_.as<VarNode>()->name_].first = ge->rhs_;
            }
            if (ge->rhs_->nodeType() == ASTNodeType::Var) {
                iters_[ge->rhs_.as<VarNode>()->name_].second = ge->lhs_;
            }
            break;
        }
        case ASTNodeType::NE: { // not EQ
            auto eq = cond.as<NENode>();
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
        elseCase = (*this)(op->elseCase_);
        iters_ = oldMap;
    }
    auto ret = makeIf(op->id(), std::move(cond), std::move(thenCase),
                      std::move(elseCase));
    ret.as<IfNode>()->infoNotCond_ = std::move(notCond);
    return ret;
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
    getHash_(op);
    return getHash_.hash().at(op);
}

Expr AnalyzeBounds::visit(const Var &_op) {
    auto __op = CompIterBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    Bound b{op}; // Don't forget itself
    updLower(op, b);
    updUpper(op, b);
    static bool inRecur = false;
    if (!inRecur) {
        inRecur = true;
        if (iters().count(op->name_)) {
            auto &&range = iters().at(op->name_);
            auto first = (*this)(range.first);
            auto second = (*this)(range.second);
            for (auto &&item : getLower(first)) {
                updLower(op, item);
            }
            for (auto &&item : getUpper(second)) {
                updUpper(op, item);
            }
        }
        inRecur = false;
    }
    return op;
}

Expr AnalyzeBounds::visit(const Load &_op) {
    auto __op = CompIterBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    Bound b{op}; // Don't forget itself
    updLower(op, b);
    updUpper(op, b);
    return op;
}

Expr AnalyzeBounds::visit(const IntConst &_op) {
    auto __op = CompIterBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IntConst);
    auto op = __op.as<IntConstNode>();
    Bound b{LinearExpr{{}, op->val_}};
    updLower(op, b);
    updUpper(op, b);
    return op;
}

Expr AnalyzeBounds::visit(const Add &_op) {
    auto __op = CompIterBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Add);
    auto op = __op.as<AddNode>();
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
    return op;
}

Expr AnalyzeBounds::visit(const Sub &_op) {
    auto __op = CompIterBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Sub);
    auto op = __op.as<SubNode>();
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
    return op;
}

Expr AnalyzeBounds::visit(const Mul &_op) {
    auto __op = CompIterBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mul);
    auto op = __op.as<MulNode>();

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

    return op;
}

Expr AnalyzeBounds::visit(const Div &_op) {
    auto __op = CompIterBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Div);
    auto op = __op.as<DivNode>();

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

    return op;
}

bool SimplifyPass::checkUpperCmp0(const Expr &normForm,
                                  const std::function<bool(int, int)> &&cmp) {
    for (auto &&upper : getUpper(normForm)) {
        if (upper.expr_->nodeType() == ASTNodeType::IntConst &&
            cmp(upper.expr_.as<IntConstNode>()->val_, 0)) {
            return true;
        }
    }
    return false;
}

bool SimplifyPass::checkLowerCmp0(const Expr &normForm,
                                  const std::function<bool(int, int)> &&cmp) {
    for (auto &&lower : getLower(normForm)) {
        if (lower.expr_->nodeType() == ASTNodeType::IntConst &&
            cmp(lower.expr_.as<IntConstNode>()->val_, 0)) {
            return true;
        }
    }
    return false;
}

Expr SimplifyPass::visit(const Div &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Div);
    auto op = __op.as<DivNode>();
    if (op->lhs_->nodeType() == ASTNodeType::IntConst &&
        op->rhs_->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(op->lhs_.as<IntConstNode>()->val_ /
                            op->rhs_.as<IntConstNode>()->val_);
    }
    return op;
}

Expr SimplifyPass::visit(const Mod &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mod);
    auto op = __op.as<DivNode>();
    if (op->lhs_->nodeType() == ASTNodeType::IntConst &&
        op->rhs_->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(op->lhs_.as<IntConstNode>()->val_ %
                            op->rhs_.as<IntConstNode>()->val_);
    }
    return op;
}

Expr SimplifyPass::visit(const Min &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Min);
    auto op = __op.as<MinNode>();

    std::function<void(const Expr &, std::unordered_set<Expr> &)> recur =
        [&recur](const Expr &expr, std::unordered_set<Expr> &list) {
            if (expr->nodeType() == ASTNodeType::Min) {
                recur(expr.as<MinNode>()->lhs_, list);
                recur(expr.as<MinNode>()->rhs_, list);
            } else {
                list.insert(expr);
            }
        };
    std::unordered_set<Expr> lhs, rhs, all;
    recur(op->lhs_, lhs);
    recur(op->rhs_, rhs);
    all.insert(lhs.begin(), lhs.end());
    all.insert(rhs.begin(), rhs.end());

    for (auto &&l : lhs) {
        for (auto &&r : rhs) {
            auto normForm = (*this)(makeSub(l, r));
            if (checkUpperCmp0(normForm, std::less_equal<int>())) {
                all.erase(r);
            } else if (checkLowerCmp0(normForm, std::greater_equal<int>())) {
                all.erase(l);
            }
        }
    }

    if (all.size() == lhs.size() + rhs.size()) {
        return op;
    } else {
        ASSERT(!all.empty());
        Expr ret;
        for (auto &&item : all) {
            ret = ret.isValid() ? makeMin(ret, item) : item;
        }
        return markMutated(ret);
    }
}

Expr SimplifyPass::visit(const Max &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Max);
    auto op = __op.as<MaxNode>();

    std::function<void(const Expr &, std::unordered_set<Expr> &)> recur =
        [&recur](const Expr &expr, std::unordered_set<Expr> &list) {
            if (expr->nodeType() == ASTNodeType::Max) {
                recur(expr.as<MaxNode>()->lhs_, list);
                recur(expr.as<MaxNode>()->rhs_, list);
            } else {
                list.insert(expr);
            }
        };
    std::unordered_set<Expr> lhs, rhs, all;
    recur(op->lhs_, lhs);
    recur(op->rhs_, rhs);
    all.insert(lhs.begin(), lhs.end());
    all.insert(rhs.begin(), rhs.end());

    for (auto &&l : lhs) {
        for (auto &&r : rhs) {
            auto normForm = (*this)(makeSub(l, r));
            if (checkUpperCmp0(normForm, std::less_equal<int>())) {
                all.erase(l);
            } else if (checkLowerCmp0(normForm, std::greater_equal<int>())) {
                all.erase(r);
            }
        }
    }

    if (all.size() == lhs.size() + rhs.size()) {
        return op;
    } else {
        ASSERT(!all.empty());
        Expr ret;
        for (auto &&item : all) {
            ret = ret.isValid() ? makeMax(ret, item) : item;
        }
        return markMutated(ret);
    }
}

Expr SimplifyPass::visit(const LT &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LT);
    auto op = __op.as<LTNode>();
    auto normForm = (*this)(makeSub(op->lhs_, op->rhs_));
    if (checkUpperCmp0(normForm, std::less<int>())) {
        return markMutated(makeIntConst(1));
    }
    if (checkLowerCmp0(normForm, std::greater_equal<int>())) {
        return markMutated(makeIntConst(0));
    }
    return op;
}

Expr SimplifyPass::visit(const LE &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LE);
    auto op = __op.as<LENode>();
    auto normForm = (*this)(makeSub(op->lhs_, op->rhs_));
    if (checkUpperCmp0(normForm, std::less_equal<int>())) {
        return markMutated(makeIntConst(1));
    }
    if (checkLowerCmp0(normForm, std::greater<int>())) {
        return markMutated(makeIntConst(0));
    }
    return op;
}

Expr SimplifyPass::visit(const GT &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::GT);
    auto op = __op.as<GTNode>();
    auto normForm = (*this)(makeSub(op->lhs_, op->rhs_));
    if (checkUpperCmp0(normForm, std::less_equal<int>())) {
        return markMutated(makeIntConst(0));
    }
    if (checkLowerCmp0(normForm, std::greater<int>())) {
        return markMutated(makeIntConst(1));
    }
    return op;
}

Expr SimplifyPass::visit(const GE &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::GE);
    auto op = __op.as<GENode>();
    auto normForm = (*this)(makeSub(op->lhs_, op->rhs_));
    if (checkUpperCmp0(normForm, std::less<int>())) {
        return markMutated(makeIntConst(0));
    }
    if (checkLowerCmp0(normForm, std::greater_equal<int>())) {
        return markMutated(makeIntConst(1));
    }
    return op;
}

Expr SimplifyPass::visit(const EQ &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::EQ);
    auto op = __op.as<EQNode>();
    auto normForm = (*this)(makeSub(op->lhs_, op->rhs_));
    if (checkUpperCmp0(normForm, std::less<int>())) {
        return markMutated(makeIntConst(0));
    }
    if (checkLowerCmp0(normForm, std::greater<int>())) {
        return markMutated(makeIntConst(0));
    }
    for (auto &&upper : getUpper(normForm)) {
        if (upper.expr_->nodeType() == ASTNodeType::IntConst &&
            upper.expr_.as<IntConstNode>()->val_ == 0) {
            for (auto &&lower : getLower(normForm)) {
                if (lower.expr_->nodeType() == ASTNodeType::IntConst &&
                    lower.expr_.as<IntConstNode>()->val_ == 0) {
                    return markMutated(makeIntConst(1));
                }
            }
        }
    }
    return op;
}

Expr SimplifyPass::visit(const NE &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::NE);
    auto op = __op.as<NENode>();
    auto normForm = (*this)(makeSub(op->lhs_, op->rhs_));
    if (checkUpperCmp0(normForm, std::less<int>())) {
        return markMutated(makeIntConst(1));
    }
    if (checkLowerCmp0(normForm, std::greater<int>())) {
        return markMutated(makeIntConst(1));
    }
    for (auto &&upper : getUpper(normForm)) {
        if (upper.expr_->nodeType() == ASTNodeType::IntConst &&
            upper.expr_.as<IntConstNode>()->val_ == 0) {
            for (auto &&lower : getLower(normForm)) {
                if (lower.expr_->nodeType() == ASTNodeType::IntConst &&
                    lower.expr_.as<IntConstNode>()->val_ == 0) {
                    return markMutated(makeIntConst(0));
                }
            }
        }
    }
    return op;
}

Expr SimplifyPass::visit(const LAnd &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LAnd);
    auto op = __op.as<LAndNode>();
    if (op->lhs_->nodeType() == ASTNodeType::IntConst) {
        return markMutated(op->lhs_.as<IntConstNode>()->val_ ? op->rhs_
                                                             : makeIntConst(0));
    }
    if (op->rhs_->nodeType() == ASTNodeType::IntConst) {
        return markMutated(op->rhs_.as<IntConstNode>()->val_ ? op->lhs_
                                                             : makeIntConst(0));
    }
    return op;
}

Expr SimplifyPass::visit(const LOr &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LOr);
    auto op = __op.as<LOrNode>();
    if (op->lhs_->nodeType() == ASTNodeType::IntConst) {
        return markMutated(op->lhs_.as<IntConstNode>()->val_ ? makeIntConst(1)
                                                             : op->rhs_);
    }
    if (op->rhs_->nodeType() == ASTNodeType::IntConst) {
        return markMutated(op->rhs_.as<IntConstNode>()->val_ ? makeIntConst(1)
                                                             : op->lhs_);
    }
    return op;
}

Expr SimplifyPass::visit(const LNot &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LNot);
    auto op = __op.as<LNotNode>();
    switch (op->expr_->nodeType()) {
    case ASTNodeType::IntConst:
        return markMutated(makeIntConst(!op->expr_.as<IntConstNode>()->val_));
    case ASTNodeType::LT:
        return markMutated(
            makeGE(op->expr_.as<LTNode>()->lhs_, op->expr_.as<LTNode>()->rhs_));
    case ASTNodeType::GT:
        return markMutated(
            makeLE(op->expr_.as<GTNode>()->lhs_, op->expr_.as<GTNode>()->rhs_));
    case ASTNodeType::LE:
        return markMutated(
            makeGT(op->expr_.as<LENode>()->lhs_, op->expr_.as<LENode>()->rhs_));
    case ASTNodeType::GE:
        return markMutated(
            makeLT(op->expr_.as<GENode>()->lhs_, op->expr_.as<GENode>()->rhs_));
    case ASTNodeType::EQ:
        return markMutated(
            makeNE(op->expr_.as<EQNode>()->lhs_, op->expr_.as<EQNode>()->rhs_));
    case ASTNodeType::NE:
        return markMutated(
            makeEQ(op->expr_.as<NENode>()->lhs_, op->expr_.as<NENode>()->rhs_));
    case ASTNodeType::LAnd:
        return markMutated(makeLOr(makeLNot(op->expr_.as<LAndNode>()->lhs_),
                                   makeLNot(op->expr_.as<LAndNode>()->rhs_)));
    case ASTNodeType::LOr:
        return markMutated(makeLAnd(makeLNot(op->expr_.as<LOrNode>()->lhs_),
                                    makeLNot(op->expr_.as<LOrNode>()->rhs_)));
    case ASTNodeType::LNot:
        return markMutated(op->expr_.as<LNotNode>()->expr_);
    default:;
    }
    return op;
}

Stmt SimplifyPass::visit(const VarDef &_op) {
    if (varScope_.count(_op->name_)) {
        throw InvalidProgram(
            "Conflict var name: " + _op->name_ +
            ". Nested vars with the same name are not allowed");
    }
    varScope_[_op->name_] = curScope_++;
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    varScope_.erase(_op->name_), curScope_--;

    if (isEmptyStmt(op->body_)) {
        return makeStmtSeq("", {});
    }

    return op;
}

Stmt SimplifyPass::visit(const For &_op) {
    if (varScope_.count(_op->iter_)) {
        throw InvalidProgram(
            "iterators with the same name in nested loops are not allowed");
    }
    varScope_[_op->iter_] = curScope_++;
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    varScope_.erase(_op->iter_), curScope_--;

    if (isEmptyStmt(op->body_)) {
        return makeStmtSeq("", {});
    }
    auto len = (*this)(makeSub(op->end_, op->begin_));
    if (len->nodeType() == ASTNodeType::IntConst) {
        auto intLen = len.as<IntConstNode>()->val_;
        if (intLen == 1) {
            return op->body_;
        }
        if (intLen <= 0) {
            return makeStmtSeq("", {});
        }
    }

    return op;
}

Stmt SimplifyPass::visit(const If &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.as<IfNode>();
    if (isEmptyStmt(op->thenCase_) && isEmptyStmt(op->elseCase_)) {
        return makeStmtSeq("", {});
    }
    if (op->cond_->nodeType() == ASTNodeType::IntConst) {
        if (op->cond_.as<IntConstNode>()->val_) {
            return markMutated(op->thenCase_);
        } else {
            if (op->elseCase_.isValid()) {
                return markMutated(op->elseCase_);
            } else {
                return markMutated(makeStmtSeq("", {}));
            }
        }
    }
    return op;
}

Stmt SimplifyPass::visit(const Assert &_op) {
    auto __op = AnalyzeBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Assert);
    auto op = __op.as<AssertNode>();
    if (op->cond_->nodeType() == ASTNodeType::IntConst) {
        if (op->cond_.as<IntConstNode>()->val_) {
            return markMutated(op->body_);
        } else {
            std::ostringstream os;
            // Print the unchanged _op
            os << "Assertion always false: " << _op;
            throw InvalidSchedule(os.str());
        }
    }
    return op;
}

void CheckFixedPoint::visitExpr(
    const Expr &op, const std::function<void(const Expr &)> &visitNode) {
    Visitor::visitExpr(op, visitNode);
    if (mutated_.count(op)) {
        isFixPoint_ = false;
    }
}

void CheckFixedPoint::visitStmt(
    const Stmt &op, const std::function<void(const Stmt &)> &visitNode) {
    Visitor::visitStmt(op, visitNode);
    if (mutated_.count(op)) {
        isFixPoint_ = false;
    }
}

Stmt simplifyPass(const Stmt &op) {
    return flattenStmtSeq(std::get<0>(simplifyAndGetBounds(op)));
}

std::tuple<Stmt, SimplifyPass::BoundsMap, SimplifyPass::BoundsMap>
simplifyAndGetBounds(const Stmt &_op) {
    auto op = _op;

    for (int i = 0;; i++) {
        op = disambiguous(op);

        SimplifyPass mutator;
        op = mutator(op);

        CheckFixedPoint checker(mutator.mutated());
        checker(op);
        if (checker.isFixPoint() || i > 100) {
            if (i > 100) {
                WARNING("SimplifyPass iterates over 100 rounds. Maybe there is "
                        "a bug");
            }
            return {op, mutator.lower(), mutator.upper()};
        }
    }
}

} // namespace ir

