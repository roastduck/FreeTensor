#include <climits>
#include <sstream>
#include <unordered_set>

#include <analyze/hash.h>
#include <except.h>
#include <math/utils.h>
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

uint64_t CompTransientBounds::getHash(const Expr &op) {
    getHash_(op);
    return getHash_.hash().at(op);
}

Expr CompTransientBounds::sub1(const Expr &op) {
    if (op->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(op.as<IntConstNode>()->val_ - 1);
    } else {
        return makeSub(op, makeIntConst(1));
    }
}

Expr CompTransientBounds::add1(const Expr &op) {
    if (op->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(op.as<IntConstNode>()->val_ + 1);
    } else {
        return makeAdd(op, makeIntConst(1));
    }
}

void CompTransientBounds::applyCond(const Expr &cond) {
    switch (cond->nodeType()) {
    case ASTNodeType::LAnd: {
        auto land = cond.as<LAndNode>();
        applyCond(land->lhs_);
        applyCond(land->rhs_);
        break;
    }
    case ASTNodeType::LT: {
        auto lt = cond.as<LTNode>();
        transients_[getHash(lt->lhs_)].second = sub1(lt->rhs_);
        transients_[getHash(lt->rhs_)].first = add1(lt->lhs_);
        break;
    }
    case ASTNodeType::GT: {
        auto gt = cond.as<GTNode>();
        transients_[getHash(gt->lhs_)].first = add1(gt->rhs_);
        transients_[getHash(gt->rhs_)].second = sub1(gt->lhs_);
        break;
    }
    case ASTNodeType::LE: {
        auto le = cond.as<LENode>();
        transients_[getHash(le->lhs_)].second = le->rhs_;
        transients_[getHash(le->rhs_)].first = le->lhs_;
        break;
    }
    case ASTNodeType::GE: {
        auto ge = cond.as<GENode>();
        transients_[getHash(ge->lhs_)].first = ge->rhs_;
        transients_[getHash(ge->rhs_)].second = ge->lhs_;
        break;
    }
    case ASTNodeType::EQ: {
        auto eq = cond.as<EQNode>();
        transients_[getHash(eq->lhs_)] = {eq->rhs_, eq->rhs_};
        transients_[getHash(eq->rhs_)] = {eq->lhs_, eq->lhs_};
        break;
    }
    default:;
        // Do nothing
    }
}

Stmt CompTransientBounds::visit(const For &op) {
    auto hash = getHash(makeVar(op->iter_));
    if (transients_.count(hash)) {
        throw InvalidProgram(
            "iterators with the same name in nested loops are not allowed");
    }
    transients_[hash] = {op->begin_, sub1(op->end_)};
    auto ret = Mutator::visit(op);
    transients_.erase(hash);
    return ret;
}

Stmt CompTransientBounds::visit(const If &op) {
    auto cond = (*this)(op->cond_);
    auto notCond = (*this)(makeLNot(cond));
    auto infoNotCond = // Different with notCond because counted in mutated_
        op->infoNotCond_.isValid() ? (*this)(op->infoNotCond_) : nullptr;

    auto oldMap = transients_;
    applyCond(cond);
    auto thenCase = (*this)(op->thenCase_);
    transients_ = oldMap;

    Stmt elseCase = nullptr;
    if (op->elseCase_.isValid()) {
        applyCond(notCond);
        elseCase = (*this)(op->elseCase_);
        transients_ = oldMap;
    }

    auto ret = makeIf(op->id(), std::move(cond), std::move(thenCase),
                      std::move(elseCase));
    ret.as<IfNode>()->infoNotCond_ = std::move(infoNotCond);
    return ret;
}

Stmt CompTransientBounds::visit(const Assert &op) {
    auto cond = (*this)(op->cond_);

    auto oldMap = transients_;
    applyCond(cond);
    auto body = (*this)(op->body_);
    transients_ = oldMap;

    return makeAssert(op->id(), std::move(cond), std::move(body));
}

std::vector<LowerBound> CompUniqueBounds::getLower(const Expr &op) const {
    if (lower_.count(op)) {
        return lower_.at(op);
    } else {
        return {};
    }
}

std::vector<UpperBound> CompUniqueBounds::getUpper(const Expr &op) const {
    if (upper_.count(op)) {
        return upper_.at(op);
    } else {
        return {};
    }
}

void CompUniqueBounds::updLower(const Expr &op, const LowerBound &bound) {
    if (!lower_.count(op)) {
        lower_[op] = {bound};
        return;
    }
    for (LowerBound &old : lower_.at(op)) {
        // The same .expr_ does not mean the same bounds
        // E.g. 1 * floor(a / 4) vs. (1/4) * a
        if (old.lin_ == bound.lin_) {
            return;
        }
        if (bound.expr_->nodeType() == ASTNodeType::IntConst &&
            old.expr_->nodeType() == ASTNodeType::IntConst) {
            auto oldVal = old.expr_.as<IntConstNode>()->val_;
            auto newVal = bound.expr_.as<IntConstNode>()->val_;
            if (newVal > oldVal) {
                old = LowerBound(LinearExpr<Rational<int>>{{}, newVal});
            }
            return;
        }
    }
    lower_.at(op).emplace_back(bound);
}

void CompUniqueBounds::updUpper(const Expr &op, const UpperBound &bound) {
    if (!upper_.count(op)) {
        upper_[op] = {bound};
        return;
    }
    for (UpperBound &old : upper_.at(op)) {
        // The same .expr_ does not mean the same bounds
        // E.g. 1 * floor(a / 4) vs. (1/4) * a
        if (old.lin_ == bound.lin_) {
            return;
        }
        if (bound.expr_->nodeType() == ASTNodeType::IntConst &&
            old.expr_->nodeType() == ASTNodeType::IntConst) {
            auto oldVal = old.expr_.as<IntConstNode>()->val_;
            auto newVal = bound.expr_.as<IntConstNode>()->val_;
            if (newVal < oldVal) {
                old = UpperBound(LinearExpr<Rational<int>>{{}, newVal});
            }
            return;
        }
    }
    upper_.at(op).emplace_back(bound);
}

int CompUniqueBounds::getIntLower(const Expr &op) const {
    int ret = INT_MIN;
    for (auto &&b : getLower(op)) {
        if (b.expr_->nodeType() == ASTNodeType::IntConst) {
            ret = std::max(ret, b.expr_.as<IntConstNode>()->val_);
        }
    }
    return ret;
}

int CompUniqueBounds::getIntUpper(const Expr &op) const {
    int ret = INT_MAX;
    for (auto &&b : getUpper(op)) {
        if (b.expr_->nodeType() == ASTNodeType::IntConst) {
            ret = std::min(ret, b.expr_.as<IntConstNode>()->val_);
        }
    }
    return ret;
}

Ref<int> CompUniqueBounds::getInt(const Expr &op) const {
    int lower = getIntLower(op);
    int upper = getIntUpper(op);
    return lower == upper ? Ref<int>::make(lower) : nullptr;
}

Expr CompUniqueBounds::visitExpr(
    const Expr &_op, const std::function<Expr(const Expr &)> &visitNode) {
    auto op = CompTransientBounds::visitExpr(_op, visitNode);
    static bool inRecur = false;
    if (!inRecur) {
        inRecur = true;
        auto hash = getHash(op);
        if (transients().count(hash)) {
            auto &&range = transients().at(hash);
            if (range.first.isValid()) {
                auto first = (*this)(range.first);
                for (auto &&item : getLower(first)) {
                    updLower(op, item);
                }
            }
            if (range.second.isValid()) {
                auto second = (*this)(range.second);
                for (auto &&item : getUpper(second)) {
                    updUpper(op, item);
                }
            }
        }
        inRecur = false;
    }
    return op;
}

Expr CompUniqueBounds::visit(const Var &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    updLower(op, LowerBound{op});
    updUpper(op, UpperBound{op});
    return op;
}

Expr CompUniqueBounds::visit(const Load &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    updLower(op, LowerBound{op});
    updUpper(op, UpperBound{op});
    return op;
}

Expr CompUniqueBounds::visit(const IntConst &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IntConst);
    auto op = __op.as<IntConstNode>();
    updLower(op, LowerBound{LinearExpr<Rational<int>>{{}, op->val_}});
    updUpper(op, UpperBound{LinearExpr<Rational<int>>{{}, op->val_}});
    return op;
}

Expr CompUniqueBounds::visit(const Add &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Add);
    auto op = __op.as<AddNode>();
    for (auto &&b1 : getLower(op->lhs_)) {
        for (auto &&b2 : getLower(op->rhs_)) {
            updLower(op, add(b1, b2));
        }
    }
    for (auto &&b1 : getUpper(op->lhs_)) {
        for (auto &&b2 : getUpper(op->rhs_)) {
            updUpper(op, add(b1, b2));
        }
    }
    return op;
}

Expr CompUniqueBounds::visit(const Sub &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Sub);
    auto op = __op.as<SubNode>();
    for (auto &&b1 : getLower(op->lhs_)) {
        for (auto &&b2 : getUpper(op->rhs_)) {
            updLower(op, sub(b1, b2));
        }
    }
    for (auto &&b1 : getUpper(op->lhs_)) {
        for (auto &&b2 : getLower(op->rhs_)) {
            updUpper(op, sub(b1, b2));
        }
    }
    return op;
}

Expr CompUniqueBounds::visit(const Mul &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mul);
    auto op = __op.as<MulNode>();

    auto g = [this](const Expr &op, const Expr &e1, const Expr &e2) {
        if (auto k = getInt(e2); k.isValid()) {
            if (*k > 0) {
                for (auto &&b : getLower(e1)) {
                    updLower(op, mul(b, *k));
                }
                for (auto &&b : getUpper(e1)) {
                    updUpper(op, mul(b, *k));
                }
            } else {
                for (auto &&b : getLower(e1)) {
                    updUpper(op, mul(UpperBound{b.lin_}, *k));
                }
                for (auto &&b : getUpper(e1)) {
                    updLower(op, mul(LowerBound{b.lin_}, *k));
                }
            }
        }
    };
    g(op, op->lhs_, op->rhs_);
    g(op, op->rhs_, op->lhs_);
    return op;
}

Expr CompUniqueBounds::visit(const FloorDiv &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::FloorDiv);
    auto op = __op.as<FloorDivNode>();

    if (auto k = getInt(op->rhs_); k.isValid()) {
        if (*k > 0) {
            for (auto &&b : getLower(op->lhs_)) {
                updLower(op, floorDiv(b, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updUpper(op, floorDiv(b, *k));
            }
        } else {
            for (auto &&b : getLower(op->lhs_)) {
                updUpper(op, floorDiv(UpperBound{b.lin_}, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updLower(op, floorDiv(LowerBound{b.lin_}, *k));
            }
        }
    }

    // Still record an monolithic item, in case floor(a / b) - floor(a / b)
    updLower(op, LowerBound{op});
    updUpper(op, UpperBound{op});

    return op;
}

Expr CompUniqueBounds::visit(const CeilDiv &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::CeilDiv);
    auto op = __op.as<CeilDivNode>();

    if (auto k = getInt(op->rhs_); k.isValid()) {
        if (*k > 0) {
            for (auto &&b : getLower(op->lhs_)) {
                updLower(op, ceilDiv(b, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updUpper(op, ceilDiv(b, *k));
            }
        } else {
            for (auto &&b : getLower(op->lhs_)) {
                updUpper(op, ceilDiv(UpperBound{b.lin_}, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updLower(op, ceilDiv(LowerBound{b.lin_}, *k));
            }
        }
    }

    // Still record an monolithic item, in case ceil(a / b) - ceil(a / b)
    updLower(op, LowerBound{op});
    updUpper(op, UpperBound{op});

    return op;
}

Expr CompUniqueBounds::visit(const Mod &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mod);
    auto op = __op.as<ModNode>();
    updLower(op, LowerBound{op});
    updUpper(op, UpperBound{op});
    updLower(op, LowerBound{LinearExpr<Rational<int>>{{}, 0}});
    for (auto &&item : getUpper(op->rhs_)) {
        updUpper(op, item);
    }
    return op;
}

Expr CompUniqueBounds::visit(const Min &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Min);
    auto op = __op.as<MinNode>();
    for (auto &&b : getUpper(op->lhs_)) {
        updUpper(op, b);
    }
    for (auto &&b : getUpper(op->rhs_)) {
        updUpper(op, b);
    }
    return op;
}

Expr CompUniqueBounds::visit(const Max &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Max);
    auto op = __op.as<MaxNode>();
    for (auto &&b : getLower(op->lhs_)) {
        updLower(op, b);
    }
    for (auto &&b : getLower(op->rhs_)) {
        updLower(op, b);
    }
    return op;
}

Expr SimplifyPass::visitExpr(
    const Expr &_op, const std::function<Expr(const Expr &)> &visitNode) {
    auto op = CompUniqueBounds::visitExpr(_op, visitNode);

    // To avoid divergence
    if (getHash(op) != getHash(_op)) {
        // E.g.
        // (1) a[0 - 0] -> a[0]
        // (2) (1 + 1) * a[0] -> 2 * a[0 - 0], because of the old bound
        return op;
    }

    Expr best = nullptr;
    auto bestScope = -1;
    for (auto &&lower : getLower(op)) {
        auto hl = getHash(lower.expr_);
        for (auto &&upper : getUpper(op)) {
            auto hr = getHash(upper.expr_);
            if (hl == hr) {
                // We need to choose the simplest one. Other wise
                // we are always picking the original expression
                auto scope = findInnerMostScope(varScope_, lower.expr_);
                if (!best.isValid() || scope < bestScope) {
                    best = lower.expr_, bestScope = scope;
                }
                break;
            }
        }
    }
    if (best.isValid() && getHash(best) != getHash(op)) {
        return markMutated(best);
    }
    return op;
}

Expr SimplifyPass::visit(const FloorDiv &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::FloorDiv);
    auto op = __op.as<FloorDivNode>();
    if (op->lhs_->nodeType() == ASTNodeType::IntConst &&
        op->rhs_->nodeType() == ASTNodeType::IntConst) {
        return markMutated(
            makeIntConst(floorDiv(op->lhs_.as<IntConstNode>()->val_,
                                  op->rhs_.as<IntConstNode>()->val_)));
    }
    return op;
}

Expr SimplifyPass::visit(const CeilDiv &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::CeilDiv);
    auto op = __op.as<CeilDivNode>();
    if (op->lhs_->nodeType() == ASTNodeType::IntConst &&
        op->rhs_->nodeType() == ASTNodeType::IntConst) {
        return markMutated(
            makeIntConst(ceilDiv(op->lhs_.as<IntConstNode>()->val_,
                                 op->rhs_.as<IntConstNode>()->val_)));
    }
    return op;
}

Expr SimplifyPass::visit(const RoundTowards0Div &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::RoundTowards0Div);
    auto op = __op.as<RoundTowards0DivNode>();
    if (op->lhs_->nodeType() == ASTNodeType::IntConst &&
        op->rhs_->nodeType() == ASTNodeType::IntConst) {
        return markMutated(makeIntConst(op->lhs_.as<IntConstNode>()->val_ /
                                        op->rhs_.as<IntConstNode>()->val_));
    }
    return op;
}

Expr SimplifyPass::visit(const Mod &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mod);
    auto op = __op.as<ModNode>();

    if (op->rhs_->nodeType() == ASTNodeType::IntConst) {
        auto k = op->rhs_.as<IntConstNode>()->val_;

        if (op->lhs_->nodeType() == ASTNodeType::IntConst) {
            return markMutated(
                makeIntConst(op->lhs_.as<IntConstNode>()->val_ % k));
        }

        bool mutated = false;
        std::function<Expr(const Expr &)> f = [&f, &mutated, k](const Expr &x) {
            switch (x->nodeType()) {
            case ASTNodeType::IntConst: {
                auto val = x.as<IntConstNode>()->val_;
                mutated = (val % k != val);
                return makeIntConst(val % k);
            }
            case ASTNodeType::Add:
                return makeAdd(f(x.as<AddNode>()->lhs_),
                               f(x.as<AddNode>()->rhs_));
            case ASTNodeType::Sub:
                return makeSub(f(x.as<SubNode>()->lhs_),
                               f(x.as<SubNode>()->rhs_));
            case ASTNodeType::Mul:
                return makeMul(f(x.as<MulNode>()->lhs_),
                               f(x.as<MulNode>()->rhs_));
            default:
                return x;
            }
        };
        auto newLhs = f(op->lhs_);
        if (mutated) {
            return markMutated(makeMod(newLhs, op->rhs_));
        }
    }

    return op;
}

Expr SimplifyPass::visit(const Min &_op) {
    auto __op = CompUniqueBounds::visit(_op);
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
            if (getIntUpper(normForm) <= 0) {
                all.erase(r);
            } else if (getIntLower(normForm) >= 0) {
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
    auto __op = CompUniqueBounds::visit(_op);
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
            if (getIntUpper(normForm) <= 0) {
                all.erase(l);
            } else if (getIntLower(normForm) >= 0) {
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
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LT);
    auto op = __op.as<LTNode>();
    auto normForm = (*this)(makeSub(op->lhs_, op->rhs_));
    if (getIntUpper(normForm) < 0) {
        return markMutated(makeBoolConst(true));
    }
    if (getIntLower(normForm) >= 0) {
        return markMutated(makeBoolConst(false));
    }
    return op;
}

Expr SimplifyPass::visit(const LE &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LE);
    auto op = __op.as<LENode>();
    auto normForm = (*this)(makeSub(op->lhs_, op->rhs_));
    if (getIntUpper(normForm) <= 0) {
        return markMutated(makeBoolConst(true));
    }
    if (getIntLower(normForm) > 0) {
        return markMutated(makeBoolConst(false));
    }
    return op;
}

Expr SimplifyPass::visit(const GT &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::GT);
    auto op = __op.as<GTNode>();
    auto normForm = (*this)(makeSub(op->lhs_, op->rhs_));
    if (getIntUpper(normForm) <= 0) {
        return markMutated(makeBoolConst(false));
    }
    if (getIntLower(normForm) > 0) {
        return markMutated(makeBoolConst(true));
    }
    return op;
}

Expr SimplifyPass::visit(const GE &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::GE);
    auto op = __op.as<GENode>();
    auto normForm = (*this)(makeSub(op->lhs_, op->rhs_));
    if (getIntUpper(normForm) < 0) {
        return markMutated(makeBoolConst(false));
    }
    if (getIntLower(normForm) >= 0) {
        return markMutated(makeBoolConst(true));
    }
    return op;
}

Expr SimplifyPass::visit(const EQ &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::EQ);
    auto op = __op.as<EQNode>();
    auto normForm = (*this)(makeSub(op->lhs_, op->rhs_));
    if (getIntUpper(normForm) < 0) {
        return markMutated(makeBoolConst(false));
    }
    if (getIntLower(normForm) > 0) {
        return markMutated(makeBoolConst(false));
    }
    if (getIntUpper(normForm) == 0 && getIntLower(normForm) == 0) {
        return markMutated(makeBoolConst(true));
    }
    return op;
}

Expr SimplifyPass::visit(const NE &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::NE);
    auto op = __op.as<NENode>();
    auto normForm = (*this)(makeSub(op->lhs_, op->rhs_));
    if (getIntUpper(normForm) < 0) {
        return markMutated(makeBoolConst(true));
    }
    if (getIntLower(normForm) > 0) {
        return markMutated(makeBoolConst(true));
    }
    if (getIntUpper(normForm) == 0 && getIntLower(normForm) == 0) {
        return markMutated(makeBoolConst(false));
    }
    return op;
}

Expr SimplifyPass::visit(const LAnd &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LAnd);
    auto op = __op.as<LAndNode>();
    if (op->lhs_->nodeType() == ASTNodeType::BoolConst) {
        return markMutated(op->lhs_.as<BoolConstNode>()->val_
                               ? op->rhs_
                               : makeBoolConst(false));
    }
    if (op->rhs_->nodeType() == ASTNodeType::BoolConst) {
        return markMutated(op->rhs_.as<BoolConstNode>()->val_
                               ? op->lhs_
                               : makeBoolConst(false));
    }
    return op;
}

Expr SimplifyPass::visit(const LOr &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LOr);
    auto op = __op.as<LOrNode>();
    if (op->lhs_->nodeType() == ASTNodeType::BoolConst) {
        return markMutated(op->lhs_.as<BoolConstNode>()->val_
                               ? makeBoolConst(true)
                               : op->rhs_);
    }
    if (op->rhs_->nodeType() == ASTNodeType::BoolConst) {
        return markMutated(op->rhs_.as<BoolConstNode>()->val_
                               ? makeBoolConst(true)
                               : op->lhs_);
    }
    return op;
}

Expr SimplifyPass::visit(const LNot &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LNot);
    auto op = __op.as<LNotNode>();
    switch (op->expr_->nodeType()) {
    case ASTNodeType::BoolConst:
        return markMutated(makeBoolConst(!op->expr_.as<BoolConstNode>()->val_));
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
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    varScope_.erase(_op->name_), curScope_--;

    if (isEmptyStmt(op->body_)) {
        return makeStmtSeq("", {});
    }

    if (op->sizeLim_.isValid()) {
        Expr size = makeIntConst(1);
        for (auto &&dim : op->buffer_->tensor().shape()) {
            size = makeMul(size, dim);
        }
        if (getIntLower((*this)(makeSub(op->sizeLim_, size))) >= 0) {
            op->sizeLim_ = nullptr;
        }
    }

    return op;
}

Stmt SimplifyPass::visit(const For &_op) {
    if (varScope_.count(_op->iter_)) {
        throw InvalidProgram(
            "iterators with the same name in nested loops are not allowed");
    }
    varScope_[_op->iter_] = curScope_++;
    auto __op = CompUniqueBounds::visit(_op);
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
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.as<IfNode>();
    if (isEmptyStmt(op->thenCase_) && isEmptyStmt(op->elseCase_)) {
        return makeStmtSeq("", {});
    }
    if (op->cond_->nodeType() == ASTNodeType::BoolConst) {
        if (op->cond_.as<BoolConstNode>()->val_) {
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
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Assert);
    auto op = __op.as<AssertNode>();
    if (op->cond_->nodeType() == ASTNodeType::BoolConst) {
        if (op->cond_.as<BoolConstNode>()->val_) {
            return markMutated(op->body_);
        } else {
            std::ostringstream os;
            // Print the unchanged _op
            os << "Assertion always false: " << _op;
            throw InvalidProgram(os.str());
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

std::tuple<Stmt, SimplifyPass::LowerBoundsMap, SimplifyPass::UpperBoundsMap>
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

