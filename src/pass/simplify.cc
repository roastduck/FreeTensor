#include <algorithm>
#include <climits>
#include <unordered_set>

#include <analyze/all_names.h>
#include <analyze/all_reads.h>
#include <analyze/all_writes.h>
#include <analyze/analyze_linear.h>
#include <analyze/as_dnf.h>
#include <except.h>
#include <math/utils.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/simplify.h>

#include "detail/simplify.h"

namespace ir {

static bool noIntersect(const std::unordered_set<std::string> &set1,
                        const std::unordered_set<std::string> &set2) {
    for (auto &&x : set1) {
        if (set2.count(x)) {
            return false;
        }
    }
    return true;
}

void FindInnerMostScope::visit(const Var &op) {
    Visitor::visit(op);
    if (!varScope_.count(op->name_)) {
        ERROR("Undefined variable: " + op->name_);
    }
    innerMost_ = std::max(innerMost_, varScope_.at(op->name_));
}

void FindInnerMostScope::visit(const Load &op) {
    Visitor::visit(op);
    if (!varScope_.count(op->var_)) {
        ERROR("Undefined variable: " + op->var_);
    }
    innerMost_ = std::max(innerMost_, varScope_.at(op->var_));
}

int findInnerMostScope(const std::unordered_map<std::string, int> &varScope,
                       const Expr &op) {
    FindInnerMostScope visitor(varScope);
    visitor(op);
    return visitor.innnerMost();
}

TransientBound CompTransientBounds::transient(const Expr &op) {
    if (transients_.count(op)) {
        return transients_.at(op);
    }
    return {};
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

void CompTransientBounds::applyCond(int k, const Expr &lhs, ASTNodeType opType,
                                    const Expr &rhs) {
    if (k < 0) {
        opType = opType == ASTNodeType::LT   ? ASTNodeType::GT
                 : opType == ASTNodeType::LE ? ASTNodeType::GE
                 : opType == ASTNodeType::GT ? ASTNodeType::LT
                 : opType == ASTNodeType::GE ? ASTNodeType::LE
                                             : opType;
        applyCond(-k, lhs, opType, makeMul(makeIntConst(-1), rhs));
        return;
    }
    auto floorRhs = k != 1 ? makeFloorDiv(rhs, makeIntConst(k)) : rhs;
    auto ceilRhs = k != 1 ? makeCeilDiv(rhs, makeIntConst(k)) : rhs;
    switch (opType) {
    case ASTNodeType::GT: {
        transients_[lhs].expr_ = (*this)(lhs);
        transients_[lhs].upper_.emplace_back((*this)(sub1(ceilRhs)));
        break;
    }
    case ASTNodeType::LT: {
        transients_[lhs].expr_ = (*this)(lhs);
        transients_[lhs].lower_.emplace_back((*this)(add1(floorRhs)));
        break;
    }
    case ASTNodeType::GE: {
        transients_[lhs].expr_ = (*this)(lhs);
        transients_[lhs].upper_.emplace_back((*this)(floorRhs));
        break;
    }
    case ASTNodeType::LE: {
        transients_[lhs].expr_ = (*this)(lhs);
        transients_[lhs].lower_.emplace_back((*this)(ceilRhs));
        break;
    }
    case ASTNodeType::EQ: {
        transients_[lhs].expr_ = (*this)(lhs);
        transients_[lhs].lower_.emplace_back((*this)(ceilRhs));
        transients_[lhs].upper_.emplace_back((*this)(floorRhs));
        break;
    }
    case ASTNodeType::NE:
        break; // do nothing
    default:
        ASSERT(false);
    }
}

void CompTransientBounds::applyCond(
    const Expr &_cond, const std::unordered_set<std::string> &bodyAllWrites) {
    auto dnf = asDNF(_cond);

    if (dnf.size() != 1) {
        return; // Currently we cannot handle OR
    }

    for (auto &&cond : dnf.front()) {
        if (!noIntersect(allReads(cond), bodyAllWrites)) {
            continue;
        }

        auto norm = linearComp(cond);
        if (!norm.isValid()) {
            continue;
        }

        auto &&[lin, type] = *norm;
        if (!isInt(dtype(lin2expr(lin)))) {
            continue;
        }

        for (auto &&[k, a] : lin.coeff_) {
            if (k != 0 && (a->nodeType() == ASTNodeType::Var ||
                           a->nodeType() == ASTNodeType::Load)) {
                auto l = lin;
                l.coeff_.resize(
                    std::remove_if(
                        l.coeff_.begin(), l.coeff_.end(),
                        [&a](const decltype(l.coeff_)::value_type &kx) {
                            return HashComparator()(kx.a_, a);
                        }) -
                    l.coeff_.begin());
                applyCond(-k, a, type, lin2expr(l));
            }
        }
        conds_.emplace_back(cond);
    }
}

Stmt CompTransientBounds::visit(const For &op) {
    auto var = makeVar(op->iter_);
    if (transients_.count(var)) {
        throw InvalidProgram(
            "iterators with the same name in nested loops are not allowed");
    }
    auto oldCondsSize = conds_.size();
    if (op->step_->nodeType() == ASTNodeType::IntConst) {
        auto step = op->step_.as<IntConstNode>()->val_;
        if (step > 0) {
            transients_[var] = {
                var, {(*this)(op->begin_)}, {(*this)(sub1(op->end_))}};
            conds_.emplace_back(makeGE(var, op->begin_));
            conds_.emplace_back(makeLT(var, op->end_));
            conds_.emplace_back(makeEQ(
                makeMod(makeSub(var, op->begin_), op->step_), makeIntConst(0)));
        } else if (step < 0) {
            transients_[var] = {
                var, {(*this)(add1(op->end_))}, {(*this)(op->begin_)}};
            conds_.emplace_back(makeLE(var, op->begin_));
            conds_.emplace_back(makeGT(var, op->end_));
            // ISL does not support negative divisor
            conds_.emplace_back(
                makeEQ(makeMod(makeSub(op->begin_, var),
                               makeSub(makeIntConst(0), op->step_)),
                       makeIntConst(0)));
        } else {
            transients_[var] = {
                var, {(*this)(op->begin_)}, {(*this)(op->begin_)}};
            conds_.emplace_back(makeEQ(var, op->begin_));
        }
    }
    auto ret = BaseClass::visit(op);
    conds_.resize(oldCondsSize);
    transients_.erase(var);
    return ret;
}

Stmt CompTransientBounds::visit(const If &op) {
    auto cond = (*this)(op->cond_);

    auto oldMap = transients_;
    auto oldCondsSize = conds_.size();
    applyCond(cond, allWrites(op->thenCase_));
    auto thenCase = (*this)(op->thenCase_);
    transients_ = oldMap;
    conds_.resize(oldCondsSize);

    Stmt elseCase = nullptr;
    if (op->elseCase_.isValid()) {
        auto oldCondsSize = conds_.size();
        applyCond(makeLNot(cond), allWrites(op->elseCase_));
        elseCase = (*this)(op->elseCase_);
        transients_ = oldMap;
        conds_.resize(oldCondsSize);
    }

    auto ret = makeIf(op->id(), std::move(cond), std::move(thenCase),
                      std::move(elseCase));
    return COPY_DEBUG_INFO(ret, op);
}

Stmt CompTransientBounds::visit(const Assert &op) {
    auto cond = (*this)(op->cond_);

    auto oldMap = transients_;
    auto oldCondsSize = conds_.size();
    applyCond(cond, allWrites(op->body_));
    auto body = (*this)(op->body_);
    transients_ = oldMap;
    conds_.resize(oldCondsSize);

    return makeAssert(op->id(), std::move(cond), std::move(body));
}

Stmt CompTransientBounds::visit(const Assume &op) {
    auto cond = (*this)(op->cond_);

    auto oldMap = transients_;
    auto oldCondsSize = conds_.size();
    applyCond(cond, allWrites(op->body_));
    auto body = (*this)(op->body_);
    transients_ = oldMap;
    conds_.resize(oldCondsSize);

    return makeAssume(op->id(), std::move(cond), std::move(body));
}

void CompUniqueBounds::updLower(LowerBoundsList &list,
                                const LowerBound &bound) const {
    for (LowerBound &old : list) {
        // The same .expr() does not mean the same bounds
        // E.g. 1 * floor(a / 4) vs. (1/4) * a
        if (old.lin() == bound.lin()) {
            return;
        }
        if (bound.lin().coeff_.empty() && old.lin().coeff_.empty()) {
            auto oldVal = old.lin().bias_;
            auto newVal = bound.lin().bias_;
            if (newVal > oldVal) {
                old = LowerBound(LinearExpr<Rational<int64_t>>{{}, newVal});
            }
            return;
        }
    }
    list.emplace_back(bound);
}

void CompUniqueBounds::updUpper(UpperBoundsList &list,
                                const UpperBound &bound) const {
    for (UpperBound &old : list) {
        // The same .expr() does not mean the same bounds
        // E.g. 1 * floor(a / 4) vs. (1/4) * a
        if (old.lin() == bound.lin()) {
            return;
        }
        if (bound.lin().coeff_.empty() && old.lin().coeff_.empty()) {
            auto oldVal = old.lin().bias_;
            auto newVal = bound.lin().bias_;
            if (newVal < oldVal) {
                old = UpperBound(LinearExpr<Rational<int64_t>>{{}, newVal});
            }
            return;
        }
    }
    list.emplace_back(bound);
}

int CompUniqueBounds::getIntLower(const Expr &op) const {
    int ret = INT_MIN;
    for (auto &&b : getLower(op)) {
        if (b.lin().coeff_.empty()) {
            auto bias = b.lin().bias_;
            ret =
                std::max(ret, (int)ceilDiv(bias.p_, bias.q_)); // FIXME: int64_t
        }
    }
    return ret;
}

int CompUniqueBounds::getIntUpper(const Expr &op) const {
    int ret = INT_MAX;
    for (auto &&b : getUpper(op)) {
        if (b.lin().coeff_.empty()) {
            auto bias = b.lin().bias_;
            ret = std::min(ret,
                           (int)floorDiv(bias.p_, bias.q_)); // FIXME: int64_t
        }
    }
    return ret;
}

Opt<int> CompUniqueBounds::getInt(const Expr &op) const {
    int lower = getIntLower(op);
    int upper = getIntUpper(op);
    return lower == upper ? Opt<int>::make(lower) : nullptr;
}

bool CompUniqueBounds::alwaysLT(const Expr &lhs, const Expr &rhs) const {
    for (auto &&b1 : getUpper(lhs)) {
        for (auto &&b2 : getLower(rhs)) {
            if (ir::alwaysLT(b1, b2)) {
                return true;
            }
        }
    }
    return false;
}

bool CompUniqueBounds::alwaysLE(const Expr &lhs, const Expr &rhs) const {
    for (auto &&b1 : getUpper(lhs)) {
        for (auto &&b2 : getLower(rhs)) {
            if (ir::alwaysLE(b1, b2)) {
                return true;
            }
        }
    }
    return false;
}

Expr CompUniqueBounds::visitExpr(const Expr &_op) {
    auto op = CompTransientBounds::visitExpr(_op);
    auto tr = transient(op);
    auto &lower = lower_[op];
    auto &upper = upper_[op];
    for (auto &&first : tr.lower_) {
        for (auto &&item : getLower(first)) {
            if (noIntersect(allNames(op), allNames(item.expr()))) {
                // No loop bounds: X cannot bound X itself
                updLower(lower, item);
            }
        }
    }
    for (auto &&second : tr.upper_) {
        for (auto &&item : getUpper(second)) {
            if (noIntersect(allNames(op), allNames(item.expr()))) {
                // No loop bounds: X cannot bound X itself
                updUpper(upper, item);
            }
        }
    }
    return op;
}

Expr CompUniqueBounds::visit(const Var &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    updLower(lower_[op], LowerBound{op});
    updUpper(upper_[op], UpperBound{op});
    return op;
}

Expr CompUniqueBounds::visit(const Load &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    if (isInt(dtype(op))) {
        updLower(lower_[op], LowerBound{op});
        updUpper(upper_[op], UpperBound{op});
    }
    return op;
}

Expr CompUniqueBounds::visit(const IntConst &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IntConst);
    auto op = __op.as<IntConstNode>();
    updLower(lower_[op],
             LowerBound{LinearExpr<Rational<int64_t>>{{}, op->val_}});
    updUpper(upper_[op],
             UpperBound{LinearExpr<Rational<int64_t>>{{}, op->val_}});
    return op;
}

Expr CompUniqueBounds::visit(const Add &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Add);
    auto op = __op.as<AddNode>();
    auto &lower = lower_[op];
    auto &upper = upper_[op];
    for (auto &&b1 : getLower(op->lhs_)) {
        for (auto &&b2 : getLower(op->rhs_)) {
            updLower(lower, add(b1, b2));
        }
    }
    for (auto &&b1 : getUpper(op->lhs_)) {
        for (auto &&b2 : getUpper(op->rhs_)) {
            updUpper(upper, add(b1, b2));
        }
    }
    return op;
}

Expr CompUniqueBounds::visit(const Sub &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Sub);
    auto op = __op.as<SubNode>();
    auto &lower = lower_[op];
    auto &upper = upper_[op];
    for (auto &&b1 : getLower(op->lhs_)) {
        for (auto &&b2 : getUpper(op->rhs_)) {
            updLower(lower, sub(b1, b2));
        }
    }
    for (auto &&b1 : getUpper(op->lhs_)) {
        for (auto &&b2 : getLower(op->rhs_)) {
            updUpper(upper, sub(b1, b2));
        }
    }
    return op;
}

Expr CompUniqueBounds::visit(const Mul &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mul);
    auto op = __op.as<MulNode>();

    auto &lower = lower_[op];
    auto &upper = upper_[op];
    auto g = [this, &lower, &upper](const Expr &op, const Expr &e1,
                                    const Expr &e2) {
        if (auto k = getInt(e2); k.isValid()) {
            if (*k > 0) {
                for (auto &&b : getLower(e1)) {
                    updLower(lower, mul(b, *k));
                }
                for (auto &&b : getUpper(e1)) {
                    updUpper(upper, mul(b, *k));
                }
                if (e1->nodeType() == ASTNodeType::FloorDiv) {
                    auto div = e1.as<FloorDivNode>();
                    if (auto k1 = getInt(div->rhs_);
                        k1.isValid() && *k1 > 0 && *k % *k1 == 0) {
                        auto equ = (*this)(
                            makeSub(div->lhs_, makeMod(div->lhs_, div->rhs_)));
                        for (auto &&b : getLower(equ)) {
                            updLower(lower, mul(b, *k / *k1));
                        }
                        for (auto &&b : getUpper(equ)) {
                            updUpper(upper, mul(b, *k / *k1));
                        }
                    }
                }
            } else {
                for (auto &&b : getLower(e1)) {
                    updUpper(upper, mul(UpperBound{b.lin()}, *k));
                }
                for (auto &&b : getUpper(e1)) {
                    updLower(lower, mul(LowerBound{b.lin()}, *k));
                }
                if (e1->nodeType() == ASTNodeType::FloorDiv) {
                    auto div = e1.as<FloorDivNode>();
                    if (auto k1 = getInt(div->rhs_);
                        k1.isValid() && *k1 > 0 && *k % *k1 == 0) {
                        auto equ = (*this)(
                            makeSub(div->lhs_, makeMod(div->lhs_, div->rhs_)));
                        for (auto &&b : getLower(equ)) {
                            updUpper(upper, mul(UpperBound{b.lin()}, *k / *k1));
                        }
                        for (auto &&b : getUpper(equ)) {
                            updLower(lower, mul(LowerBound{b.lin()}, *k / *k1));
                        }
                    }
                }
            }
        }
    };
    g(op, op->lhs_, op->rhs_);
    g(op, op->rhs_, op->lhs_);
    return op;
}

Expr CompUniqueBounds::visit(const Square &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Square);
    auto op = __op.as<SquareNode>();

    auto &lower = lower_[op];
    auto &upper = upper_[op];
    if (auto k = getInt(op->expr_); k.isValid()) {
        updLower(lower, LowerBound{LinearExpr<Rational<int64_t>>{{}, *k * *k}});
        updUpper(upper, UpperBound{LinearExpr<Rational<int64_t>>{{}, *k * *k}});
    }
    return op;
}

Expr CompUniqueBounds::visit(const FloorDiv &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::FloorDiv);
    auto op = __op.as<FloorDivNode>();

    auto &lower = lower_[op];
    auto &upper = upper_[op];
    if (auto k = getInt(op->rhs_); k.isValid()) {
        if (*k > 0) {
            for (auto &&b : getLower(op->lhs_)) {
                updLower(lower, floorDiv(b, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updUpper(upper, floorDiv(b, *k));
            }
        } else {
            for (auto &&b : getLower(op->lhs_)) {
                updUpper(upper, floorDiv(UpperBound{b.lin()}, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updLower(lower, floorDiv(LowerBound{b.lin()}, *k));
            }
        }
    }

    return op;
}

Expr CompUniqueBounds::visit(const CeilDiv &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::CeilDiv);
    auto op = __op.as<CeilDivNode>();

    auto &lower = lower_[op];
    auto &upper = upper_[op];
    if (auto k = getInt(op->rhs_); k.isValid()) {
        if (*k > 0) {
            for (auto &&b : getLower(op->lhs_)) {
                updLower(lower, ceilDiv(b, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updUpper(upper, ceilDiv(b, *k));
            }
        } else {
            for (auto &&b : getLower(op->lhs_)) {
                updUpper(upper, ceilDiv(UpperBound{b.lin()}, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updLower(lower, ceilDiv(LowerBound{b.lin()}, *k));
            }
        }
    }

    return op;
}

Expr CompUniqueBounds::visit(const Mod &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mod);
    auto op = __op.as<ModNode>();
    updLower(lower_[op], LowerBound{op});
    updUpper(upper_[op], UpperBound{op});
    updLower(lower_[op], LowerBound{LinearExpr<Rational<int64_t>>{{}, 0}});
    for (auto &&item : getUpper(op->rhs_)) {
        updUpper(upper_[op], item);
    }
    return op;
}

Expr CompUniqueBounds::visit(const Min &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Min);
    auto op = __op.as<MinNode>();

    if (!isInt(dtype(op))) {
        return op;
    }
    auto &lower = lower_[op];
    auto &upper = upper_[op];
    for (auto &&b : getUpper(op->lhs_)) {
        updUpper(upper, b);
    }
    for (auto &&b : getUpper(op->rhs_)) {
        updUpper(upper, b);
    }
    for (auto &&b1 : getLower(op->lhs_)) {
        for (auto &&b2 : getLower(op->rhs_)) {
            if (b1.lin().coeff_.empty() && b2.lin().coeff_.empty()) {
                updLower(lower,
                         LinearExpr<Rational<int64_t>>{
                             {}, std::min(b1.lin().bias_, b2.lin().bias_)});
            }
        }
    }
    updLower(lower, LowerBound{op});
    updUpper(upper, UpperBound{op});
    return op;
}

Expr CompUniqueBounds::visit(const Max &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Max);
    auto op = __op.as<MaxNode>();

    if (!isInt(dtype(op))) {
        return op;
    }
    auto &lower = lower_[op];
    auto &upper = upper_[op];
    for (auto &&b : getLower(op->lhs_)) {
        updLower(lower, b);
    }
    for (auto &&b : getLower(op->rhs_)) {
        updLower(lower, b);
    }
    for (auto &&b1 : getUpper(op->lhs_)) {
        for (auto &&b2 : getUpper(op->rhs_)) {
            if (b1.lin().coeff_.empty() && b2.lin().coeff_.empty()) {
                updUpper(upper,
                         LinearExpr<Rational<int64_t>>{
                             {}, std::max(b1.lin().bias_, b2.lin().bias_)});
            }
        }
    }
    updLower(lower, LowerBound{op});
    updUpper(upper, UpperBound{op});
    return op;
}

Expr CompUniqueBounds::visit(const IfExpr &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IfExpr);
    auto op = __op.as<IfExprNode>();

    if (!isInt(dtype(op))) {
        return op;
    }
    auto &lower = lower_[op];
    auto &upper = upper_[op];
    for (auto &&b1 : getUpper(op->thenCase_)) {
        for (auto &&b2 : getUpper(op->elseCase_)) {
            if (b1.lin().coeff_.empty() && b2.lin().coeff_.empty()) {
                updUpper(upper,
                         LinearExpr<Rational<int64_t>>{
                             {}, std::max(b1.lin().bias_, b2.lin().bias_)});
            }
        }
    }
    for (auto &&b1 : getLower(op->thenCase_)) {
        for (auto &&b2 : getLower(op->elseCase_)) {
            if (b1.lin().coeff_.empty() && b2.lin().coeff_.empty()) {
                updLower(lower,
                         LinearExpr<Rational<int64_t>>{
                             {}, std::min(b1.lin().bias_, b2.lin().bias_)});
            }
        }
    }
    updLower(lower, LowerBound{op});
    updUpper(upper, UpperBound{op});
    return op;
}

Stmt builtinSimplify(const Stmt &op) {
    return flattenStmtSeq(
        std::get<0>(simplifyAndGetBounds<BuiltinSimplify>(op)));
}

Stmt simplifyPass(const Stmt &op) { return builtinSimplify(op); }

} // namespace ir
