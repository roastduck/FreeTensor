#include <algorithm>
#include <unordered_set>

#include <analyze/as_dnf.h>
#include <except.h>
#include <math/min_max.h>
#include <math/utils.h>
#include <pass/annotate_conds.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/replace_iter.h>
#include <pass/simplify.h>

namespace freetensor {

static std::vector<Expr> factorize(const Expr &expr) {
    std::vector<Expr> factors;
    std::function<void(const Expr &)> recur = [&](const Expr &expr) {
        if (expr->nodeType() == ASTNodeType::Mul) {
            recur(expr.as<MulNode>()->lhs_);
            recur(expr.as<MulNode>()->rhs_);
        } else {
            factors.emplace_back(expr);
        }
    };
    recur(expr);
    return factors;
}

static std::tuple<std::vector<Expr>, std::vector<Expr>, bool>
factorDiv(const std::vector<Expr> &_lhs, const std::vector<Expr> &_rhs) {
    // We use vector scan instead of hash set here to ensure deterministisy
    auto lhs = _lhs, rhs = _rhs;
    bool modified = false;
    for (auto i = rhs.begin(); i != rhs.end();) {
        for (auto j = lhs.begin(); j != lhs.end();) {
            if (HashComparator{}(*j, *i)) {
                i = rhs.erase(i);
                j = lhs.erase(j);
                modified = true;
                goto next;
            } else {
                j++;
            }
        }
        i++;
    next:;
    }
    return {lhs, rhs, modified};
}

static Expr reduceMul(const std::vector<Expr> &factors) {
    Expr ret;
    for (auto &&item : factors) {
        ret = ret.isValid() ? makeMul(ret, item) : item;
    }
    return ret.isValid() ? ret : makeIntConst(1);
}

static Expr recursiveNegateMul(const Expr &e) {
    if (e->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(-e.as<IntConstNode>()->val_);
    } else if (e->nodeType() == ASTNodeType::FloatConst) {
        return makeFloatConst(-e.as<FloatConstNode>()->val_);
    } else if (e->nodeType() == ASTNodeType::Mul) {
        auto &&mul = e.as<MulNode>();
        if (auto &&nl = recursiveNegateMul(mul->lhs_); nl.isValid()) {
            return makeMul(nl, mul->rhs_);
        } else if (auto &&nr = recursiveNegateMul(mul->rhs_); nr.isValid()) {
            return makeMul(mul->lhs_, nr);
        } else {
            return nullptr;
        }
    } else {
        return nullptr;
    }
}

static std::pair<Expr, int64_t> recursiveGetConstOffset(const Expr &e) {
    if (e->nodeType() == ASTNodeType::IntConst) {
        return {nullptr, e.as<IntConstNode>()->val_};
    } else if (e->nodeType() == ASTNodeType::Add) {
        auto &&add = e.as<AddNode>();
        auto &&[le, lc] = recursiveGetConstOffset(add->lhs_);
        auto &&[re, rc] = recursiveGetConstOffset(add->rhs_);
        return {le.isValid() && re.isValid() ? makeAdd(le, re)
                : le.isValid()               ? le
                                             : re,
                lc + rc};
    } else if (e->nodeType() == ASTNodeType::Sub) {
        auto &&sub = e.as<SubNode>();
        auto &&[le, lc] = recursiveGetConstOffset(sub->lhs_);
        auto &&[re, rc] = recursiveGetConstOffset(sub->rhs_);
        return {le.isValid() && re.isValid() ? makeSub(le, re)
                : le.isValid()               ? le
                                             : makeSub(makeIntConst(0), re),
                lc - rc};
    } else {
        return {e, 0};
    }
}

namespace {

class CountExprNodes : public Visitor {
    int count_ = 0;

  public:
    int count() const { return count_; }

  protected:
    void visitExpr(const Expr &e) {
        Visitor::visitExpr(e);
        count_++;
    }
};

int countExprNodes(const Expr &e) {
    CountExprNodes visitor;
    visitor(e);
    return visitor.count();
}

} // Anonymous namespace

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

Stmt SimplifyPass::visitStmt(const Stmt &op) {
    auto uniqueOfOuterStmt = unique_;
    unique_ = compUniqueBoundsFactory_(*this);
    auto ret = BaseClass::visitStmt(op);
    unique_ = uniqueOfOuterStmt;
    return ret;
}

Expr SimplifyPass::visitExpr(const Expr &_op) {
    if (isInt(_op->dtype())) {
        if (leafFirstBoundAnalysis_) {
            auto op = BaseClass::visitExpr(_op);
            if (!HashComparator()(op, _op)) {
                // To avoid divergence
                // E.g.
                // (1) a[0 - 0] -> a[0]
                // (2) (1 + 1) * a[0] -> 2 * a[0 - 0], because of the old bound
                return op;
            }
            if (auto bound = unique_->getBound(op); bound.isValid()) {
                Expr best = bound->simplestExpr(op, varScope_);
                if (best.isValid() && !HashComparator()(best, op)) {
                    return best;
                }
            }
            return op;
        } else {
            if (auto bound = unique_->getBound(_op); bound.isValid()) {
                Expr best = bound->simplestExpr(_op, varScope_);
                if (best.isValid() && !HashComparator()(best, _op)) {
                    return best;
                }
            }
            return BaseClass::visitExpr(_op);
        }

    } else if (isBool(_op->dtype())) {
        auto op = BaseClass::visitExpr(_op);
        if (auto p = _op->parentExpr();
            !p.isValid() || p->dtype() != DataType::Bool) {
            // this is base bool expr
            Expr normalized = makeLOrLAnd(asDNF(op));
            if (countExprNodes(normalized) < countExprNodes(op)) {
                op = normalized;
            }
        }
        return op;

    } else {
        return BaseClass::visitExpr(_op);
    }
}

Expr SimplifyPass::visit(const Add &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Add);
    auto op = __op.as<AddNode>();

    if (equals(op->lhs_, 0)) {
        return op->rhs_;
    }
    if (equals(op->rhs_, 0)) {
        return op->lhs_;
    }

    if (op->lhs_->nodeType() == ASTNodeType::IntConst) {
        if (auto &&[re, rc] = recursiveGetConstOffset(op->rhs_); rc != 0) {
            return makeAdd(makeIntConst(op->lhs_.as<IntConstNode>()->val_ + rc),
                           re);
        }
    }
    if (op->rhs_->nodeType() == ASTNodeType::IntConst) {
        if (auto &&[le, lc] = recursiveGetConstOffset(op->lhs_); lc != 0) {
            return makeAdd(
                le, makeIntConst(op->rhs_.as<IntConstNode>()->val_ + lc));
        }
    }

    if (op->lhs_->isConst() && op->rhs_->nodeType() == ASTNodeType::Min) {
        return makeMin(makeAdd(op->lhs_, op->rhs_.as<MinNode>()->lhs_),
                       makeAdd(op->lhs_, op->rhs_.as<MinNode>()->rhs_));
    }
    if (op->lhs_->isConst() && op->rhs_->nodeType() == ASTNodeType::Max) {
        return makeMax(makeAdd(op->lhs_, op->rhs_.as<MaxNode>()->lhs_),
                       makeAdd(op->lhs_, op->rhs_.as<MaxNode>()->rhs_));
    }
    if (op->lhs_->nodeType() == ASTNodeType::Min && op->rhs_->isConst()) {
        return makeMin(makeAdd(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                       makeAdd(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
    }
    if (op->lhs_->nodeType() == ASTNodeType::Max && op->rhs_->isConst()) {
        return makeMax(makeAdd(op->lhs_.as<MaxNode>()->lhs_, op->rhs_),
                       makeAdd(op->lhs_.as<MaxNode>()->rhs_, op->rhs_));
    }

    return op;
}

Expr SimplifyPass::visit(const Sub &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Sub);
    auto op = __op.as<SubNode>();

    if (equals(op->lhs_, 0)) {
        if (auto &&nr = recursiveNegateMul(op->rhs_); nr.isValid()) {
            return nr;
        }
    }
    if (equals(op->rhs_, 0)) {
        return op->lhs_;
    }

    if (op->lhs_->nodeType() == ASTNodeType::IntConst) {
        if (auto &&[re, rc] = recursiveGetConstOffset(op->rhs_); rc != 0) {
            return makeSub(makeIntConst(op->lhs_.as<IntConstNode>()->val_ - rc),
                           re);
        }
    }
    if (op->rhs_->nodeType() == ASTNodeType::IntConst) {
        if (auto &&[le, lc] = recursiveGetConstOffset(op->lhs_); lc != 0) {
            return makeAdd(
                le, makeIntConst(lc - op->rhs_.as<IntConstNode>()->val_));
        }
    }

    if (op->lhs_->isConst() && op->rhs_->nodeType() == ASTNodeType::Min) {
        return makeMax(makeSub(op->lhs_, op->rhs_.as<MinNode>()->lhs_),
                       makeSub(op->lhs_, op->rhs_.as<MinNode>()->rhs_));
    }
    if (op->lhs_->isConst() && op->rhs_->nodeType() == ASTNodeType::Max) {
        return makeMin(makeSub(op->lhs_, op->rhs_.as<MinNode>()->lhs_),
                       makeSub(op->lhs_, op->rhs_.as<MinNode>()->rhs_));
    }
    if (op->lhs_->nodeType() == ASTNodeType::Min && op->rhs_->isConst()) {
        return makeMin(makeSub(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                       makeSub(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
    }
    if (op->lhs_->nodeType() == ASTNodeType::Max && op->rhs_->isConst()) {
        return makeMax(makeSub(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                       makeSub(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
    }

    return op;
}

Expr SimplifyPass::visit(const Mul &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Mul);
    auto op = __op.as<MulNode>();

    if (equals(op->lhs_, 1)) {
        return op->rhs_;
    }
    if (equals(op->rhs_, 1)) {
        return op->lhs_;
    }
    if (equals(op->lhs_, 0)) {
        return makeIntConst(0);
    }
    if (equals(op->rhs_, 0)) {
        return makeIntConst(0);
    }
    if (equals(op->lhs_, -1)) {
        if (auto &&nr = recursiveNegateMul(op->rhs_); nr.isValid()) {
            return nr;
        }
    }
    if (equals(op->rhs_, -1)) {
        if (auto &&nl = recursiveNegateMul(op->lhs_); nl.isValid()) {
            return nl;
        }
    }

    if (op->lhs_->isConst() && op->rhs_->nodeType() == ASTNodeType::Min) {
        if (isGT0(op->lhs_->dtype())) {
            return makeMin(makeMul(op->lhs_, op->rhs_.as<MinNode>()->lhs_),
                           makeMul(op->lhs_, op->rhs_.as<MinNode>()->rhs_));
        }
        if (isLT0(op->lhs_->dtype())) {
            return makeMax(makeMul(op->lhs_, op->rhs_.as<MinNode>()->lhs_),
                           makeMul(op->lhs_, op->rhs_.as<MinNode>()->rhs_));
        }
    }
    if (op->lhs_->isConst() && op->rhs_->nodeType() == ASTNodeType::Max) {
        if (isGT0(op->lhs_->dtype())) {
            return makeMax(makeMul(op->lhs_, op->rhs_.as<MinNode>()->lhs_),
                           makeMul(op->lhs_, op->rhs_.as<MinNode>()->rhs_));
        }
        if (isLT0(op->lhs_->dtype())) {
            return makeMin(makeMul(op->lhs_, op->rhs_.as<MinNode>()->lhs_),
                           makeMul(op->lhs_, op->rhs_.as<MinNode>()->rhs_));
        }
    }
    if (op->lhs_->nodeType() == ASTNodeType::Min && op->rhs_->isConst()) {
        if (isGT0(op->rhs_->dtype())) {
            return makeMin(makeMul(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                           makeMul(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
        if (isLT0(op->rhs_->dtype())) {
            return makeMax(makeMul(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                           makeMul(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
    }
    if (op->lhs_->nodeType() == ASTNodeType::Max && op->rhs_->isConst()) {
        if (isGT0(op->rhs_->dtype())) {
            return makeMax(makeMul(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                           makeMul(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
        if (isLT0(op->rhs_->dtype())) {
            return makeMin(makeMul(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                           makeMul(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
    }

    return op;
}

Expr SimplifyPass::visit(const FloorDiv &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::FloorDiv);
    auto op = __op.as<FloorDivNode>();
    if (auto &&[factorsL, factorsR, modified] =
            factorDiv(factorize(op->lhs_), factorize(op->rhs_));
        modified) {
        // floor(a * b / a * c) == floor(b / c)
        op = makeFloorDiv(reduceMul(factorsL), reduceMul(factorsR))
                 .as<FloorDivNode>();
    }
    if (equals(op->rhs_, 1)) {
        return op->lhs_;
    }
    if (equals(op->rhs_, -1)) {
        return makeMul(makeIntConst(-1), op->lhs_);
    }

    if (op->lhs_->nodeType() == ASTNodeType::Min && op->rhs_->isConst()) {
        if (isGT0(op->rhs_->dtype())) {
            return makeMin(
                makeFloorDiv(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                makeFloorDiv(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
        if (isLT0(op->rhs_->dtype())) {
            return makeMax(
                makeFloorDiv(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                makeFloorDiv(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
    }
    if (op->lhs_->nodeType() == ASTNodeType::Max && op->rhs_->isConst()) {
        if (isGT0(op->rhs_->dtype())) {
            return makeMax(
                makeFloorDiv(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                makeFloorDiv(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
        if (isLT0(op->rhs_->dtype())) {
            return makeMin(
                makeFloorDiv(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                makeFloorDiv(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
    }

    return op;
}

Expr SimplifyPass::visit(const CeilDiv &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::CeilDiv);
    auto op = __op.as<CeilDivNode>();
    if (auto &&[factorsL, factorsR, modified] =
            factorDiv(factorize(op->lhs_), factorize(op->rhs_));
        modified) {
        // ceil(a * b / a * c) == ceil(b / c)
        op = makeCeilDiv(reduceMul(factorsL), reduceMul(factorsR))
                 .as<CeilDivNode>();
    }
    if (equals(op->rhs_, 1)) {
        return op->lhs_;
    }
    if (equals(op->rhs_, -1)) {
        return makeMul(makeIntConst(-1), op->lhs_);
    }

    if (op->lhs_->nodeType() == ASTNodeType::Min && op->rhs_->isConst()) {
        if (isGT0(op->rhs_->dtype())) {
            return makeMin(makeCeilDiv(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                           makeCeilDiv(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
        if (isLT0(op->rhs_->dtype())) {
            return makeMax(makeCeilDiv(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                           makeCeilDiv(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
    }
    if (op->lhs_->nodeType() == ASTNodeType::Max && op->rhs_->isConst()) {
        if (isGT0(op->rhs_->dtype())) {
            return makeMax(makeCeilDiv(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                           makeCeilDiv(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
        if (isLT0(op->rhs_->dtype())) {
            return makeMin(makeCeilDiv(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                           makeCeilDiv(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
    }

    return op;
}

Expr SimplifyPass::visit(const RoundTowards0Div &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::RoundTowards0Div);
    auto op = __op.as<RoundTowards0DivNode>();
    if (auto &&[factorsL, factorsR, modified] =
            factorDiv(factorize(op->lhs_), factorize(op->rhs_));
        modified) {
        // round(a * b / a * c) == round(b / c)
        op = makeRoundTowards0Div(reduceMul(factorsL), reduceMul(factorsR))
                 .as<RoundTowards0DivNode>();
    }
    if (equals(op->rhs_, 1)) {
        return op->lhs_;
    }
    if (equals(op->rhs_, -1)) {
        return makeMul(makeIntConst(-1), op->lhs_);
    }

    if (op->lhs_->nodeType() == ASTNodeType::Min && op->rhs_->isConst()) {
        if (isGT0(op->rhs_->dtype())) {
            return makeMin(
                makeRoundTowards0Div(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                makeRoundTowards0Div(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
        if (isLT0(op->rhs_->dtype())) {
            return makeMax(
                makeRoundTowards0Div(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                makeRoundTowards0Div(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
    }
    if (op->lhs_->nodeType() == ASTNodeType::Max && op->rhs_->isConst()) {
        if (isGT0(op->rhs_->dtype())) {
            return makeMax(
                makeRoundTowards0Div(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                makeRoundTowards0Div(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
        if (isLT0(op->rhs_->dtype())) {
            return makeMin(
                makeRoundTowards0Div(op->lhs_.as<MinNode>()->lhs_, op->rhs_),
                makeRoundTowards0Div(op->lhs_.as<MinNode>()->rhs_, op->rhs_));
        }
    }

    return op;
}

Expr SimplifyPass::visit(const Mod &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Mod);
    auto op = __op.as<ModNode>();

    if (auto &&[factorsL, factorsR, modified] =
            factorDiv(factorize(op->lhs_), factorize(op->rhs_));
        factorsR.empty()) {
        // (n * m) % n == 0, but (n * m) % (n * k) !== m % k
        return makeIntConst(0);
    }

    if (unique_->getIntLower(op->rhs_) > 0 &&
        unique_->getIntLower(op->lhs_) >= 0 &&
        unique_->alwaysLT(op->lhs_, op->rhs_)) {
        return op->lhs_;
    }
    if (unique_->getIntUpper(op->rhs_) < 0 &&
        unique_->getIntUpper(op->rhs_) <= 0 &&
        unique_->alwaysLT(op->rhs_, op->lhs_)) {
        return op->lhs_;
    }

    if (op->rhs_->nodeType() == ASTNodeType::IntConst) {
        auto k = op->rhs_.as<IntConstNode>()->val_;

        if (k == 1 || k == -1) {
            return makeIntConst(0);
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
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Remainder);
    auto op = __op.as<RemainderNode>();
    if (auto &&[factorsL, factorsR, modified] =
            factorDiv(factorize(op->lhs_), factorize(op->rhs_));
        factorsR.empty()) {
        // (n * m) %% n == 0, but (n * m) %% (n * k) !== m %% k
        return makeIntConst(0);
    }
    return op;
}

Expr SimplifyPass::visit(const LT &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::LT);
    auto op = __op.as<LTNode>();
    if (!isInt(op->lhs_->dtype()) || !isInt(op->rhs_->dtype())) {
        return op;
    }
    auto diff = makeSub(op->lhs_, op->rhs_);
    if (unique_->getIntUpper(diff) < 0) {
        return makeBoolConst(true);
    }
    if (unique_->getIntLower(diff) >= 0) {
        return makeBoolConst(false);
    }
    return op;
}

Expr SimplifyPass::visit(const LE &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::LE);
    auto op = __op.as<LENode>();
    if (!isInt(op->lhs_->dtype()) || !isInt(op->rhs_->dtype())) {
        return op;
    }
    auto diff = makeSub(op->lhs_, op->rhs_);
    if (unique_->getIntUpper(diff) <= 0) {
        return makeBoolConst(true);
    }
    if (unique_->getIntLower(diff) > 0) {
        return makeBoolConst(false);
    }
    return op;
}

Expr SimplifyPass::visit(const GT &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::GT);
    auto op = __op.as<GTNode>();
    if (!isInt(op->lhs_->dtype()) || !isInt(op->rhs_->dtype())) {
        return op;
    }
    auto diff = makeSub(op->lhs_, op->rhs_);
    if (unique_->getIntLower(diff) > 0) {
        return makeBoolConst(true);
    }
    if (unique_->getIntUpper(diff) <= 0) {
        return makeBoolConst(false);
    }
    return op;
}

Expr SimplifyPass::visit(const GE &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::GE);
    auto op = __op.as<GENode>();
    if (!isInt(op->lhs_->dtype()) || !isInt(op->rhs_->dtype())) {
        return op;
    }
    auto diff = makeSub(op->lhs_, op->rhs_);
    if (unique_->getIntLower(diff) >= 0) {
        return makeBoolConst(true);
    }
    if (unique_->getIntUpper(diff) < 0) {
        return makeBoolConst(false);
    }
    return op;
}

Expr SimplifyPass::visit(const EQ &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::EQ);
    auto op = __op.as<EQNode>();
    if (!isInt(op->lhs_->dtype()) || !isInt(op->rhs_->dtype())) {
        return op;
    }
    auto diff = makeSub(op->lhs_, op->rhs_);
    if (unique_->getIntLower(diff) > 0) {
        return makeBoolConst(false);
    }
    if (unique_->getIntUpper(diff) < 0) {
        return makeBoolConst(false);
    }
    if (auto &&c = unique_->getInt(diff); c.has_value() && *c == 0) {
        return makeBoolConst(true);
    }
    return op;
}

Expr SimplifyPass::visit(const NE &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::NE);
    auto op = __op.as<NENode>();
    if (!isInt(op->lhs_->dtype()) || !isInt(op->rhs_->dtype())) {
        return op;
    }
    auto diff = makeSub(op->lhs_, op->rhs_);
    if (unique_->getIntLower(diff) > 0) {
        return makeBoolConst(true);
    }
    if (unique_->getIntUpper(diff) < 0) {
        return makeBoolConst(true);
    }
    if (auto &&c = unique_->getInt(diff); c.has_value() && *c == 0) {
        return makeBoolConst(false);
    }
    return op;
}

Expr SimplifyPass::visit(const IfExpr &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::IfExpr);
    auto op = __op.as<IfExprNode>();

    if (op->cond_->nodeType() == ASTNodeType::BoolConst) {
        if (op->cond_.as<BoolConstNode>()->val_) {
            return op->thenCase_;
        } else {
            return op->elseCase_;
        }
    }

    if (HashComparator{}(op->thenCase_, op->elseCase_)) {
        return op->thenCase_;
    }

    if (op->thenCase_->nodeType() == op->elseCase_->nodeType()) {
        if (op->thenCase_->isUnary()) {
            return makeUnary(
                op->thenCase_->nodeType(),
                makeIfExpr(op->cond_, op->thenCase_.as<UnaryExprNode>()->expr_,
                           op->elseCase_.as<UnaryExprNode>()->expr_));
        } else if (op->thenCase_->isBinary()) {
            auto &&thenCase = op->thenCase_.as<BinaryExprNode>();
            auto &&elseCase = op->elseCase_.as<BinaryExprNode>();
            if (HashComparator{}(thenCase->lhs_, elseCase->lhs_)) {
                return makeBinary(
                    thenCase->nodeType(), thenCase->lhs_,
                    makeIfExpr(op->cond_, thenCase->rhs_, elseCase->rhs_));
            } else if (HashComparator{}(thenCase->rhs_, elseCase->rhs_)) {
                return makeBinary(
                    thenCase->nodeType(),
                    makeIfExpr(op->cond_, thenCase->lhs_, elseCase->lhs_),
                    thenCase->rhs_);
            } else if (thenCase->isCommutative()) {
                if (HashComparator{}(thenCase->lhs_, elseCase->rhs_)) {
                    return makeBinary(
                        thenCase->nodeType(), thenCase->lhs_,
                        makeIfExpr(op->cond_, thenCase->rhs_, elseCase->lhs_));
                } else if (HashComparator{}(thenCase->rhs_, elseCase->lhs_)) {
                    return makeBinary(
                        thenCase->nodeType(), thenCase->rhs_,
                        makeIfExpr(op->cond_, thenCase->lhs_, elseCase->rhs_));
                }
            }
        } else if (op->thenCase_->nodeType() == ASTNodeType::Cast) {
            auto &&thenCase = op->thenCase_.as<CastNode>();
            auto &&elseCase = op->elseCase_.as<CastNode>();
            if (thenCase->destType_ == elseCase->destType_) {
                return makeCast(
                    makeIfExpr(op->cond_, thenCase->expr_, elseCase->expr_),
                    thenCase->destType_);
            }
        } else if (op->thenCase_->nodeType() == ASTNodeType::Load) {
            auto &&thenCase = op->thenCase_.as<LoadNode>();
            auto &&elseCase = op->elseCase_.as<LoadNode>();
            if (thenCase->var_ == elseCase->var_) {
                // Since `var_` is the same, these must be the same
                ASSERT(thenCase->indices_.size() == elseCase->indices_.size());
                ASSERT(thenCase->loadType_ == elseCase->loadType_);
                int diffCnt = 0;
                std::vector<Expr> indices;
                indices.reserve(thenCase->indices_.size());
                for (auto &&[thenItem, elseItem] :
                     views::zip(thenCase->indices_, elseCase->indices_)) {
                    if (HashComparator{}(thenItem, elseItem)) {
                        indices.emplace_back(thenItem);
                    } else {
                        diffCnt++;
                        indices.emplace_back(
                            makeIfExpr(op->cond_, thenItem, elseItem));
                    }
                }
                if (diffCnt <= 1) {
                    return makeLoad(thenCase->var_, std::move(indices),
                                    thenCase->loadType_);
                }
            }
        }
        // TODO: We can also handle `Intrinsic`, but we must properly deal with
        // `hasSideEffect_`, and check for data type in case of `... ?
        // intrinsic(..., type_a) : intrinsic(..., type_b)`.
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
            return makeStmtSeq({});
        }
        if (op->expr_->nodeType() == ASTNodeType::FloatConst &&
            op->expr_.as<FloatConstNode>()->val_ == 0) {
            return makeStmtSeq({});
        }
        break;
    case ReduceOp::Mul:
        if (op->expr_->nodeType() == ASTNodeType::IntConst &&
            op->expr_.as<IntConstNode>()->val_ == 1) {
            return makeStmtSeq({});
        }
        if (op->expr_->nodeType() == ASTNodeType::FloatConst &&
            op->expr_.as<FloatConstNode>()->val_ == 1) {
            return makeStmtSeq({});
        }
        break;
    case ReduceOp::LAnd:
        if (op->expr_->nodeType() == ASTNodeType::BoolConst &&
            op->expr_.as<BoolConstNode>()->val_ == true) {
            return makeStmtSeq({});
        }
        break;
    case ReduceOp::LOr:
        if (op->expr_->nodeType() == ASTNodeType::BoolConst &&
            op->expr_.as<BoolConstNode>()->val_ == false) {
            return makeStmtSeq({});
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

    if (auto _intLen = unique_->getInt(op->len_); _intLen.has_value()) {
        auto intLen = *_intLen;
        if (intLen == 1) {
            auto body = ReplaceIter(_op->iter_, op->begin_)(_op->body_);
            return (*this)(body);
        }
        if (intLen <= 0) {
            return makeStmtSeq({});
        }
    }
    if (unique_->getIntUpper(op->len_) == 1) {
        auto body = ReplaceIter(_op->iter_, op->begin_)(_op->body_);
        body = (*this)(body);
        return makeIf(makeEQ(op->len_, makeIntConst(1)), body);
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
                return makeStmtSeq({});
            }
        }
    }

    return BaseClass::visit(makeIf(std::move(cond), _op->thenCase_,
                                   _op->elseCase_, _op->metadata(), _op->id())
                                .as<IfNode>());
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
            throw AssertAlwaysFalse(FT_MSG << "Assertion always false: "
                                           << _op);
        }
    }
    return op;
}

template <class Simplifier> static Stmt simplifyImpl(const Stmt &_op) {
    auto op = _op;

    for (int i = 0;; i++) {
        auto newOp = annotateConds(op);
        newOp = Simplifier()(newOp);
        newOp = flattenStmtSeq(newOp);
        if (HashComparator()(newOp, op) || i > 100) {
            if (i > 100) {
                WARNING("pass/simplify iterates over 100 rounds. Maybe there "
                        "is a bug");
            }
            return newOp;
        }
        op = newOp;
    }
}

Stmt builtinSimplify(const Stmt &op) {
    return flattenStmtSeq(simplifyImpl<BuiltinSimplify>(op));
}

Stmt pbSimplify(const Stmt &op) {
    return flattenStmtSeq(simplifyImpl<PBSimplify>(op));
}

Stmt simplify(const Stmt &op) { return builtinSimplify(op); }

} // namespace freetensor
