#include <climits>
#include <unordered_set>

#include <analyze/hash.h>
#include <except.h>
#include <math/utils.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/simplify.h>

#include "detail/simplify.h"

namespace ir {

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

uint64_t FindBoundAccess::getHash(const Expr &op) {
    getHash_(op);
    return getHash_.hash().at(op);
}

void FindBoundAccess::setBoundAccess(const Expr &op) {
    if (dontInsert_)
        return;
    switch (op->nodeType()) {
    case ASTNodeType::Var: {
        auto hash = getHash(op.as<VarNode>());
        boundAccess_.emplace(hash);
        break;
    }
    case ASTNodeType::Load: {
        auto hash = getHash(makeVar(op.as<LoadNode>()->var_));
        boundAccess_.emplace(hash);
        break;
    }
    default:;
        // Do nothing
    }
}

bool FindBoundAccess::checkBoundAccess(const Expr &op) {
    switch (op->nodeType()) {
    case ASTNodeType::Var: {
        auto hash = getHash(op);
        return boundAccess_.count(hash);
    }
    case ASTNodeType::Load: {
        auto hash = getHash(makeVar(op.as<LoadNode>()->var_));
        return boundAccess_.count(hash);
    }
    default: {
        return true;
    }
    }
}

Stmt FindBoundAccess::visit(const Store &op) {
    auto hash = getHash(makeVar(op->var_));
    boundAccess_.erase(hash);
    auto ret = Mutator::visit(op);
    return ret;
}

Stmt FindBoundAccess::visit(const ReduceTo &op) {
    auto hash = getHash(makeVar(op->var_));
    boundAccess_.erase(hash);
    auto ret = Mutator::visit(op);
    return ret;
}

std::pair<std::vector<Expr>, std::vector<Expr>>
CompTransientBounds::transient(const Expr &op) {
    // FIXME: Currently we only check if X is out-of-date in A <= X <= B, but
    // not in f(X) <= Y <= g(X)
    auto hash = getHash(op);
    if (transients_.count(hash)) {
        if (!checkBoundAccess(op)) {
            transients_.erase(hash);
        }
        if (transients_.count(hash)) {
            return transients_.at(hash);
        }
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
    case ASTNodeType::LT: {
        transients_[getHash(lhs)].second.emplace_back(sub1(ceilRhs));
        setBoundAccess(lhs);
        break;
    }
    case ASTNodeType::GT: {
        transients_[getHash(lhs)].first.emplace_back(add1(floorRhs));
        setBoundAccess(lhs);
        break;
    }
    case ASTNodeType::LE: {
        transients_[getHash(lhs)].second.emplace_back(floorRhs);
        setBoundAccess(lhs);
        break;
    }
    case ASTNodeType::GE: {
        transients_[getHash(lhs)].first.emplace_back(ceilRhs);
        setBoundAccess(lhs);
        break;
    }
    case ASTNodeType::EQ: {
        transients_[getHash(lhs)].first.emplace_back(ceilRhs);
        transients_[getHash(lhs)].second.emplace_back(floorRhs);
        setBoundAccess(lhs);
        break;
    }
    default:
        ASSERT(false);
    }
}

void CompTransientBounds::applyCond(const Expr &cond) {
    Expr norm;
    switch (cond->nodeType()) {
    case ASTNodeType::LAnd: {
        auto land = cond.as<LAndNode>();
        applyCond(land->lhs_);
        applyCond(land->rhs_);
        return;
    }
    case ASTNodeType::LT: {
        auto lt = cond.as<LTNode>();
        norm = makeSub(lt->rhs_, lt->lhs_);
        break;
    }
    case ASTNodeType::GT: {
        auto gt = cond.as<GTNode>();
        norm = makeSub(gt->rhs_, gt->lhs_);
        break;
    }
    case ASTNodeType::LE: {
        auto le = cond.as<LENode>();
        norm = makeSub(le->rhs_, le->lhs_);
        break;
    }
    case ASTNodeType::GE: {
        auto ge = cond.as<GENode>();
        norm = makeSub(ge->rhs_, ge->lhs_);
        break;
    }
    case ASTNodeType::EQ: {
        auto eq = cond.as<EQNode>();
        norm = makeSub(eq->rhs_, eq->lhs_);
        break;
    }
    default:
        return;
    }

    analyzeLinear_(norm);
    if (!analyzeLinear_.result().count(norm)) {
        return;
    }
    LinearExpr lin = analyzeLinear_.result().at(norm);
    for (auto &&item : lin.coeff_) {
        if (item.second.k_ != 0 &&
            (item.second.a_->nodeType() == ASTNodeType::Var ||
             item.second.a_->nodeType() == ASTNodeType::Load)) {
            auto l = lin;
            l.coeff_.erase(item.first);
            applyCond(-item.second.k_, item.second.a_, cond->nodeType(),
                      lin2expr(l));
        }
    }
}

Stmt CompTransientBounds::visit(const For &op) {
    FindBoundAccess fbaccess;
    fbaccess.dontInsert();
    fbaccess.boundAccess(boundAccess_);
    fbaccess.findBoundAccess(op);
    boundAccess_ = fbaccess.boundAccess();
    auto hash = getHash(makeVar(op->iter_));
    if (transients_.count(hash)) {
        throw InvalidProgram(
            "iterators with the same name in nested loops are not allowed");
    }
    transients_[hash] = {{op->begin_}, {sub1(op->end_)}};
    boundAccess_.emplace(hash);
    auto ret = FindBoundAccess::visit(op);
    transients_.erase(hash);
    return ret;
}

Stmt CompTransientBounds::visit(const If &op) {
    auto cond = (*this)(op->cond_);
    auto notCond = (*this)(makeLNot(cond));

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
    return COPY_DEBUG_INFO(ret, op);
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
        auto tr = transient(op);
        for (auto &&_first : tr.first) {
            auto first = (*this)(_first);
            for (auto &&item : getLower(first)) {
                updLower(op, item);
            }
        }
        for (auto &&_second : tr.second) {
            auto second = (*this)(_second);
            for (auto &&item : getUpper(second)) {
                updUpper(op, item);
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
    for (auto &&b1 : getLower(op->lhs_)) {
        for (auto &&b2 : getLower(op->rhs_)) {
            if (b1.lin_.coeff_.empty() && b2.lin_.coeff_.empty()) {
                updLower(op, LinearExpr<Rational<int>>{
                                 {}, std::min(b1.lin_.bias_, b2.lin_.bias_)});
            }
        }
    }
    updLower(op, LowerBound{op});
    updUpper(op, UpperBound{op});
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
    for (auto &&b1 : getUpper(op->lhs_)) {
        for (auto &&b2 : getUpper(op->rhs_)) {
            if (b1.lin_.coeff_.empty() && b2.lin_.coeff_.empty()) {
                updUpper(op, LinearExpr<Rational<int>>{
                                 {}, std::max(b1.lin_.bias_, b2.lin_.bias_)});
            }
        }
    }
    updLower(op, LowerBound{op});
    updUpper(op, UpperBound{op});
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

Stmt builtinSimplify(const Stmt &op) {
    return flattenStmtSeq(
        std::get<0>(simplifyAndGetBounds<BuiltinSimplify>(op)));
}

Stmt simplifyPass(const Stmt &op) { return builtinSimplify(op); }

} // namespace ir
