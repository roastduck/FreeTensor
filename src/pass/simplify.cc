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
#include <pass/replace_iter.h>
#include <pass/simplify.h>

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

class CountHeavyOps : public Visitor {
    int cnt_ = 0;

  public:
    int cnt() const { return cnt_; }

  private:
    template <class T> void visitHeavy(const T &op) {
        Visitor::visit(op);
        cnt_++;
    }

  protected:
    void visit(const RealDiv &op) override { visitHeavy(op); }
    void visit(const FloorDiv &op) override { visitHeavy(op); }
    void visit(const CeilDiv &op) override { visitHeavy(op); }
    void visit(const RoundTowards0Div &op) override { visitHeavy(op); }
    void visit(const Mod &op) override { visitHeavy(op); }
    void visit(const Sqrt &op) override { visitHeavy(op); }
    void visit(const Exp &op) override { visitHeavy(op); }
};

static int countHeavyOps(const Expr &op) {
    CountHeavyOps visitor;
    visitor(op);
    return visitor.cnt();
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

TransientBound CompTransientBounds::transient(const Expr &op) const {
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
            if (a->nodeType() == ASTNodeType::Var ||
                a->nodeType() == ASTNodeType::Load) {
                auto [lower, upper] = lin2bounds(lin, type, a);
                if (lower.isValid()) {
                    transients_[a].expr_ = a;
                    transients_[a].lower_.emplace_back(lower->expr());
                }
                if (upper.isValid()) {
                    transients_[a].expr_ = a;
                    transients_[a].upper_.emplace_back(upper->expr());
                }
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
            transients_[var] = {var, {op->begin_}, {sub1(op->end_)}};
            conds_.emplace_back(makeGE(var, op->begin_));
            conds_.emplace_back(makeLT(var, op->end_));
            conds_.emplace_back(makeEQ(
                makeMod(makeSub(var, op->begin_), op->step_), makeIntConst(0)));
        } else if (step < 0) {
            transients_[var] = {var, {add1(op->end_)}, {op->begin_}};
            conds_.emplace_back(makeLE(var, op->begin_));
            conds_.emplace_back(makeGT(var, op->end_));
            // ISL does not support negative divisor
            conds_.emplace_back(
                makeEQ(makeMod(makeSub(op->begin_, var),
                               makeSub(makeIntConst(0), op->step_)),
                       makeIntConst(0)));
        } else {
            transients_[var] = {var, {op->begin_}, {op->begin_}};
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

int CompUniqueBounds::getIntLower(const Expr &op) {
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

int CompUniqueBounds::getIntUpper(const Expr &op) {
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

Opt<int> CompUniqueBounds::getInt(const Expr &op) {
    int lower = getIntLower(op);
    int upper = getIntUpper(op);
    return lower == upper ? Opt<int>::make(lower) : nullptr;
}

bool CompUniqueBounds::alwaysLT(const Expr &lhs, const Expr &rhs) {
    for (auto &&b1 : getUpper(lhs)) {
        for (auto &&b2 : getLower(rhs)) {
            if (ir::alwaysLT(b1, b2)) {
                return true;
            }
        }
    }
    return false;
}

bool CompUniqueBounds::alwaysLE(const Expr &lhs, const Expr &rhs) {
    for (auto &&b1 : getUpper(lhs)) {
        for (auto &&b2 : getLower(rhs)) {
            if (ir::alwaysLE(b1, b2)) {
                return true;
            }
        }
    }
    return false;
}

void CompUniqueBounds::visitExpr(const Expr &op) {
    if (lower_.count(op) || upper_.count(op)) {
        return;
    }
    auto &lower = lower_[op];
    auto &upper = upper_[op];
    lower = {};
    upper = {};

    if (!isInt(dtype(op))) {
        return;
    }

    BaseClass::visitExpr(op);
    auto tr = transients_.transient(op);
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
}

void CompUniqueBounds::visit(const Var &op) {
    BaseClass::visit(op);
    updLower(lower_[op], LowerBound{op});
    updUpper(upper_[op], UpperBound{op});
}

void CompUniqueBounds::visit(const Load &op) {
    BaseClass::visit(op);
    updLower(lower_[op], LowerBound{op});
    updUpper(upper_[op], UpperBound{op});
}

void CompUniqueBounds::visit(const IntConst &op) {
    BaseClass::visit(op);
    updLower(lower_[op],
             LowerBound{LinearExpr<Rational<int64_t>>{{}, op->val_}});
    updUpper(upper_[op],
             UpperBound{LinearExpr<Rational<int64_t>>{{}, op->val_}});
}

void CompUniqueBounds::visit(const Add &op) {
    BaseClass::visit(op);
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
}

void CompUniqueBounds::visit(const Sub &op) {
    BaseClass::visit(op);
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
}

void CompUniqueBounds::visit(const Mul &op) {
    BaseClass::visit(op);

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
                        auto equ =
                            makeSub(div->lhs_, makeMod(div->lhs_, div->rhs_));
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
                        auto equ =
                            makeSub(div->lhs_, makeMod(div->lhs_, div->rhs_));
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
}

void CompUniqueBounds::visit(const Square &op) {
    BaseClass::visit(op);

    auto &lower = lower_[op];
    auto &upper = upper_[op];
    if (auto k = getInt(op->expr_); k.isValid()) {
        updLower(lower, LowerBound{LinearExpr<Rational<int64_t>>{{}, *k * *k}});
        updUpper(upper, UpperBound{LinearExpr<Rational<int64_t>>{{}, *k * *k}});
    }
}

void CompUniqueBounds::visit(const FloorDiv &op) {
    BaseClass::visit(op);

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
}

void CompUniqueBounds::visit(const CeilDiv &op) {
    BaseClass::visit(op);

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
}

void CompUniqueBounds::visit(const Mod &op) {
    BaseClass::visit(op);
    updLower(lower_[op], LowerBound{op});
    updUpper(upper_[op], UpperBound{op});
    updLower(lower_[op], LowerBound{LinearExpr<Rational<int64_t>>{{}, 0}});
    for (auto &&item : getUpper(op->rhs_)) {
        updUpper(upper_[op], item);
    }
}

void CompUniqueBounds::visit(const Min &op) {
    BaseClass::visit(op);
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
}

void CompUniqueBounds::visit(const Max &op) {
    BaseClass::visit(op);
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
}

void CompUniqueBounds::visit(const IfExpr &op) {
    BaseClass::visit(op);
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
}

Expr SimplifyPass::visitExpr(const Expr &_op) {
    auto op = BaseClass::visitExpr(_op);

    // To avoid divergence
    if (!HashComparator()(op, _op)) {
        // E.g.
        // (1) a[0 - 0] -> a[0]
        // (2) (1 + 1) * a[0] -> 2 * a[0 - 0], because of the old bound
        return op;
    }

    Expr best = nullptr;
    auto bestScope = -1, bestHeavyOps = -1;
    for (auto &&lower : unique_.getLower(op)) {
        for (auto &&upper : unique_.getUpper(op)) {
            if (ir::alwaysLE(upper, lower)) { // upper <= lower ==> equal
                // We need to choose the simplest one. Otherwise we are always
                // picking the original expression
                Expr expr;
                if (upper.lin().coeff_.size() + (upper.lin().bias_ != 0) >
                    lower.lin().coeff_.size() + (lower.lin().bias_ != 0)) {
                    expr = lower.expr();
                } else {
                    expr = upper.expr();
                }
                auto scope = findInnerMostScope(varScope_, expr);
                auto heavyOps = countHeavyOps(expr);
                if (!best.isValid() || scope < bestScope ||
                    (scope == bestScope && heavyOps < bestHeavyOps)) {
                    best = expr, bestScope = scope, bestHeavyOps = heavyOps;
                }
                break;
            }
        }
    }
    if (best.isValid() && !HashComparator()(best, op)) {
        return best;
    }
    return op;
}

Expr SimplifyPass::visit(const IntConst &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IntConst);
    auto op = __op.as<IntConstNode>();
    constants_[op] = op->val_;
    return op;
}

Expr SimplifyPass::visit(const Add &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Add);
    auto op = __op.as<AddNode>();

    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return makeIntConst(constants_.at(op->lhs_) + constants_.at(op->rhs_));
    }
    if (constants_.count(op->lhs_) && constants_.at(op->lhs_) == 0) {
        return op->rhs_;
    }
    if (constants_.count(op->rhs_) && constants_.at(op->rhs_) == 0) {
        return op->lhs_;
    }

    return op;
}

Expr SimplifyPass::visit(const Sub &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Sub);
    auto op = __op.as<SubNode>();

    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return makeIntConst(constants_.at(op->lhs_) - constants_.at(op->rhs_));
    }
    if (constants_.count(op->rhs_) && constants_.at(op->rhs_) == 0) {
        return op->lhs_;
    }

    return op;
}

Expr SimplifyPass::visit(const Mul &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mul);
    auto op = __op.as<MulNode>();

    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return makeIntConst(constants_.at(op->lhs_) * constants_.at(op->rhs_));
    }
    if (constants_.count(op->lhs_) && constants_.at(op->lhs_) == 1) {
        return op->rhs_;
    }
    if (constants_.count(op->rhs_) && constants_.at(op->rhs_) == 1) {
        return op->lhs_;
    }
    if (constants_.count(op->lhs_) && constants_.at(op->lhs_) == 0) {
        return makeIntConst(0);
    }
    if (constants_.count(op->rhs_) && constants_.at(op->rhs_) == 0) {
        return makeIntConst(0);
    }

    return op;
}

Expr SimplifyPass::visit(const FloorDiv &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::FloorDiv);
    auto op = __op.as<FloorDivNode>();
    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return makeIntConst(
            floorDiv(constants_.at(op->lhs_), constants_.at(op->rhs_)));
    }
    return op;
}

Expr SimplifyPass::visit(const CeilDiv &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::CeilDiv);
    auto op = __op.as<CeilDivNode>();
    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return makeIntConst(
            ceilDiv(constants_.at(op->lhs_), constants_.at(op->rhs_)));
    }
    return op;
}

Expr SimplifyPass::visit(const RoundTowards0Div &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::RoundTowards0Div);
    auto op = __op.as<RoundTowards0DivNode>();
    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return makeIntConst(constants_.at(op->lhs_) / constants_.at(op->rhs_));
    }
    return op;
}

Expr SimplifyPass::visit(const Mod &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mod);
    auto op = __op.as<ModNode>();

    if (unique_.getIntLower(op->rhs_) > 0 &&
        unique_.getIntLower(op->lhs_) >= 0 &&
        unique_.alwaysLT(op->lhs_, op->rhs_)) {
        return op->lhs_;
    }
    if (unique_.getIntUpper(op->rhs_) < 0 &&
        unique_.getIntUpper(op->rhs_) <= 0 &&
        unique_.alwaysLT(op->rhs_, op->lhs_)) {
        return op->lhs_;
    }

    if (constants_.count(op->rhs_)) {
        auto k = constants_.at(op->rhs_);

        if (constants_.count(op->lhs_)) {
            return makeIntConst(mod(constants_.at(op->lhs_), k));
        }

        bool mutated = false;
        std::function<Expr(const Expr &)> f = [&f, &mutated, k](const Expr &x) {
            switch (x->nodeType()) {
            case ASTNodeType::IntConst: {
                auto val = x.as<IntConstNode>()->val_;
                mutated = (mod(val, k) != val);
                return makeIntConst(mod(val, k));
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
            return makeMod(newLhs, op->rhs_);
        }
    }

    return op;
}

Expr SimplifyPass::visit(const Remainder &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Remainder);
    auto op = __op.as<RemainderNode>();

    if (constants_.count(op->rhs_) && constants_.count(op->lhs_)) {
        return makeIntConst(constants_.at(op->lhs_) % constants_.at(op->rhs_));
    }

    return op;
}

Expr SimplifyPass::visit(const Min &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Min);
    auto op = __op.as<MinNode>();

    // Followed by rules only for integers
    if (!isInt(dtype(op))) {
        return op;
    }

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
            if (unique_.alwaysLE(l, r)) {
                all.erase(r);
            } else if (unique_.alwaysLE(r, l)) {
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
        return ret;
    }
}

Expr SimplifyPass::visit(const Max &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Max);
    auto op = __op.as<MaxNode>();

    // Followed by rules only for integers
    if (!isInt(dtype(op))) {
        return op;
    }

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
            if (unique_.alwaysLE(l, r)) {
                all.erase(l);
            } else if (unique_.alwaysLE(r, l)) {
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
        return ret;
    }
}

Expr SimplifyPass::visit(const LT &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LT);
    auto op = __op.as<LTNode>();
    if (!isInt(dtype(op->lhs_)) || !isInt(dtype(op->rhs_))) {
        return op;
    }
    if (unique_.alwaysLT(op->lhs_, op->rhs_)) {
        return makeBoolConst(true);
    }
    if (unique_.alwaysLE(op->rhs_, op->lhs_)) {
        return makeBoolConst(false);
    }
    return op;
}

Expr SimplifyPass::visit(const LE &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LE);
    auto op = __op.as<LENode>();
    if (!isInt(dtype(op->lhs_)) || !isInt(dtype(op->rhs_))) {
        return op;
    }
    if (unique_.alwaysLE(op->lhs_, op->rhs_)) {
        return makeBoolConst(true);
    }
    if (unique_.alwaysLT(op->rhs_, op->lhs_)) {
        return makeBoolConst(false);
    }
    return op;
}

Expr SimplifyPass::visit(const GT &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::GT);
    auto op = __op.as<GTNode>();
    if (!isInt(dtype(op->lhs_)) || !isInt(dtype(op->rhs_))) {
        return op;
    }
    if (unique_.alwaysLE(op->lhs_, op->rhs_)) {
        return makeBoolConst(false);
    }
    if (unique_.alwaysLT(op->rhs_, op->lhs_)) {
        return makeBoolConst(true);
    }
    return op;
}

Expr SimplifyPass::visit(const GE &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::GE);
    auto op = __op.as<GENode>();
    if (!isInt(dtype(op->lhs_)) || !isInt(dtype(op->rhs_))) {
        return op;
    }
    if (unique_.alwaysLT(op->lhs_, op->rhs_)) {
        return makeBoolConst(false);
    }
    if (unique_.alwaysLE(op->rhs_, op->lhs_)) {
        return makeBoolConst(true);
    }
    return op;
}

Expr SimplifyPass::visit(const EQ &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::EQ);
    auto op = __op.as<EQNode>();
    if (!isInt(dtype(op->lhs_)) || !isInt(dtype(op->rhs_))) {
        return op;
    }
    if (unique_.alwaysLT(op->lhs_, op->rhs_)) {
        return makeBoolConst(false);
    }
    if (unique_.alwaysLT(op->rhs_, op->lhs_)) {
        return makeBoolConst(false);
    }
    if (unique_.alwaysLE(op->lhs_, op->rhs_) &&
        unique_.alwaysLE(op->rhs_, op->lhs_)) {
        return makeBoolConst(true);
    }
    return op;
}

Expr SimplifyPass::visit(const NE &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::NE);
    auto op = __op.as<NENode>();
    if (!isInt(dtype(op->lhs_)) || !isInt(dtype(op->rhs_))) {
        return op;
    }
    if (unique_.alwaysLT(op->lhs_, op->rhs_)) {
        return makeBoolConst(true);
    }
    if (unique_.alwaysLT(op->rhs_, op->lhs_)) {
        return makeBoolConst(true);
    }
    if (unique_.alwaysLE(op->lhs_, op->rhs_) &&
        unique_.alwaysLE(op->rhs_, op->lhs_)) {
        return makeBoolConst(false);
    }
    return op;
}

Expr SimplifyPass::visit(const LAnd &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LAnd);
    auto op = __op.as<LAndNode>();
    if (op->lhs_->nodeType() == ASTNodeType::BoolConst) {
        return op->lhs_.as<BoolConstNode>()->val_ ? (Expr)op->rhs_
                                                  : makeBoolConst(false);
    }
    if (op->rhs_->nodeType() == ASTNodeType::BoolConst) {
        return op->rhs_.as<BoolConstNode>()->val_ ? (Expr)op->lhs_
                                                  : makeBoolConst(false);
    }
    return op;
}

Expr SimplifyPass::visit(const LOr &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LOr);
    auto op = __op.as<LOrNode>();
    if (op->lhs_->nodeType() == ASTNodeType::BoolConst) {
        return op->lhs_.as<BoolConstNode>()->val_ ? makeBoolConst(true)
                                                  : (Expr)op->rhs_;
    }
    if (op->rhs_->nodeType() == ASTNodeType::BoolConst) {
        return op->rhs_.as<BoolConstNode>()->val_ ? makeBoolConst(true)
                                                  : (Expr)op->lhs_;
    }
    return op;
}

Expr SimplifyPass::visit(const LNot &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LNot);
    auto op = __op.as<LNotNode>();
    switch (op->expr_->nodeType()) {
    case ASTNodeType::BoolConst:
        return makeBoolConst(!op->expr_.as<BoolConstNode>()->val_);
    case ASTNodeType::LT:
        return makeGE(op->expr_.as<LTNode>()->lhs_,
                      op->expr_.as<LTNode>()->rhs_);
    case ASTNodeType::GT:
        return makeLE(op->expr_.as<GTNode>()->lhs_,
                      op->expr_.as<GTNode>()->rhs_);
    case ASTNodeType::LE:
        return makeGT(op->expr_.as<LENode>()->lhs_,
                      op->expr_.as<LENode>()->rhs_);
    case ASTNodeType::GE:
        return makeLT(op->expr_.as<GENode>()->lhs_,
                      op->expr_.as<GENode>()->rhs_);
    case ASTNodeType::EQ:
        return makeNE(op->expr_.as<EQNode>()->lhs_,
                      op->expr_.as<EQNode>()->rhs_);
    case ASTNodeType::NE:
        return makeEQ(op->expr_.as<NENode>()->lhs_,
                      op->expr_.as<NENode>()->rhs_);
    case ASTNodeType::LAnd:
        return makeLOr(makeLNot(op->expr_.as<LAndNode>()->lhs_),
                       makeLNot(op->expr_.as<LAndNode>()->rhs_));
    case ASTNodeType::LOr:
        return makeLAnd(makeLNot(op->expr_.as<LOrNode>()->lhs_),
                        makeLNot(op->expr_.as<LOrNode>()->rhs_));
    case ASTNodeType::LNot:
        return op->expr_.as<LNotNode>()->expr_;
    default:;
    }
    return op;
}

Expr SimplifyPass::visit(const IfExpr &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IfExpr);
    auto op = __op.as<IfExprNode>();
    if (op->cond_->nodeType() == ASTNodeType::BoolConst) {
        if (op->cond_.as<BoolConstNode>()->val_) {
            return op->thenCase_;
        } else {
            return op->elseCase_;
        }
    }
    return op;
}

Stmt SimplifyPass::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    switch (op->op_) {
    case ReduceOp::Add:
        if (op->expr_->nodeType() == ASTNodeType::IntConst &&
            op->expr_.as<IntConstNode>()->val_ == 0) {
            return makeStmtSeq("", {});
        }
        if (op->expr_->nodeType() == ASTNodeType::FloatConst &&
            op->expr_.as<FloatConstNode>()->val_ == 0) {
            return makeStmtSeq("", {});
        }
        break;
    case ReduceOp::Mul:
        if (op->expr_->nodeType() == ASTNodeType::IntConst &&
            op->expr_.as<IntConstNode>()->val_ == 1) {
            return makeStmtSeq("", {});
        }
        if (op->expr_->nodeType() == ASTNodeType::FloatConst &&
            op->expr_.as<FloatConstNode>()->val_ == 1) {
            return makeStmtSeq("", {});
        }
        break;
    default:; // do nothing
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
    auto __op = BaseClass::visit(_op);
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
        if (unique_.getIntLower(makeSub(op->sizeLim_, size)) >= 0) {
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
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    varScope_.erase(_op->iter_), curScope_--;

    if (auto intLen_ = unique_.getInt(op->len_); intLen_.isValid()) {
        auto intLen = *intLen_;
        if (intLen == 1) {
            auto body = ReplaceIter(_op->iter_, op->begin_)(_op->body_);
            return (*this)(body);
        }
        if (intLen <= 0) {
            return makeStmtSeq("", {});
        }
    }
    if (unique_.getIntUpper(op->len_) == 1) {
        auto body = ReplaceIter(_op->iter_, op->begin_)(_op->body_);
        body = (*this)(body);
        return makeIf("", makeEQ(op->len_, makeIntConst(1)), body);
    }

    if (isEmptyStmt(op->body_)) {
        return makeStmtSeq("", {});
    }
    return op;
}

Stmt SimplifyPass::visit(const If &_op) {
    // Simplify the condition first to determine a possible dead branch, so we
    // can avoid recurse into the dead branch. This allows assertion false in
    // the dead branch
    auto cond = (*this)(_op->cond_);
    if (cond->nodeType() == ASTNodeType::BoolConst) {
        if (cond.as<BoolConstNode>()->val_) {
            return (*this)(_op->thenCase_);
        } else {
            if (_op->elseCase_.isValid()) {
                return (*this)(_op->elseCase_);
            } else {
                return makeStmtSeq("", {});
            }
        }
    }

    auto __op = BaseClass::visit(
        makeIf(_op->id(), std::move(cond), _op->thenCase_, _op->elseCase_)
            .as<IfNode>());
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.as<IfNode>();
    bool emptyThen = isEmptyStmt(op->thenCase_);
    bool emptyElse = isEmptyStmt(op->elseCase_);
    if (emptyThen && emptyElse) {
        return makeStmtSeq("", {});
    }
    if (op->elseCase_.isValid()) {
        if (emptyThen) {
            return makeIf(op->id(), makeLNot(op->cond_), op->elseCase_);
        }
        if (emptyElse) {
            op->elseCase_ = nullptr;
        }
    }
    return op;
}

Stmt SimplifyPass::visit(const Assert &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Assert);
    auto op = __op.as<AssertNode>();
    if (op->cond_->nodeType() == ASTNodeType::BoolConst) {
        if (op->cond_.as<BoolConstNode>()->val_) {
            return op->body_;
        } else {
            // Print the unchanged _op
            throw AssertAlwaysFalse("Assertion always false: " + toString(_op));
        }
    }
    return op;
}

Stmt builtinSimplify(const Stmt &op) {
    return flattenStmtSeq(
        std::get<0>(simplifyAndGetBounds<BuiltinSimplify>(op)));
}

Stmt simplifyPass(const Stmt &op) { return builtinSimplify(op); }

} // namespace ir
