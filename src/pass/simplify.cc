#include <algorithm>
#include <climits>
#include <unordered_set>

#include <iostream>

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

void CheckBoundOutDated::visit(const Var &op) {
    if (op.as<VarNode>()->name_ == name_)
        outDated_ = true;
    return;
}

void CheckBoundOutDated::visit(const Load &op) {
    if (op.as<LoadNode>()->var_ == name_)
        outDated_ = true;
    return;
}

void OutDatedBoundsRemover::remove(const std::string &name) {
    check_ = CheckBoundOutDated(name);
    for (auto &item : transients_) {
        check_.reset();
        check_(item.second.expr_);
        if (check_.isoutDated()) {
            item.second.lower_ = item.second.upper_ = {};
        }
        for (auto i = item.second.lower_.begin();
             i != item.second.lower_.end();) {
            check_.reset(), check_(*i);
            if (check_.isoutDated()) {
                item.second.lower_.erase(i);
            } else
                i++;
        }
        for (auto i = item.second.upper_.begin();
             i != item.second.upper_.end();) {
            check_.reset(), check_(*i);
            if (check_.isoutDated()) {
                i = item.second.upper_.erase(i);
            } else
                i++;
        }
    }
}

void OutDatedBoundsRemover::visit(const Store &op) {
    Visitor::visit(op);
    remove(op->var_);
}

void OutDatedBoundsRemover::visit(const ReduceTo &op) {
    Visitor::visit(op);
    remove(op->var_);
}

DataType CompTransientBounds::dtype(const Expr &op) {
    typeInfer_(op);
    return typeInfer_.types().at(op);
}

uint64_t CompTransientBounds::getHash(const Expr &op) {
    getHash_(op);
    return getHash_.hash().at(op);
}

TransientBound CompTransientBounds::transient(const Expr &op) {
    auto hash = getHash(op);
    if (transients_.count(hash)) {
        return transients_.at(hash);
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
    auto h = getHash(lhs);
    switch (opType) {
    case ASTNodeType::LT: {
        transients_[h].expr_ = lhs;
        transients_[h].upper_.emplace_back(sub1(ceilRhs));
        break;
    }
    case ASTNodeType::GT: {
        transients_[h].expr_ = lhs;
        transients_[h].lower_.emplace_back(add1(floorRhs));
        break;
    }
    case ASTNodeType::LE: {
        transients_[h].expr_ = lhs;
        transients_[h].upper_.emplace_back(floorRhs);
        break;
    }
    case ASTNodeType::GE: {
        transients_[h].expr_ = lhs;
        transients_[h].lower_.emplace_back(ceilRhs);
        break;
    }
    case ASTNodeType::EQ: {
        transients_[h].expr_ = lhs;
        transients_[h].lower_.emplace_back(ceilRhs);
        transients_[h].upper_.emplace_back(floorRhs);
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

    if (!isInt(dtype(norm))) {
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
            l.coeff_.resize(std::remove_if(l.coeff_.begin(), l.coeff_.end(),
                                           [&item](const decltype(
                                               l.coeff_)::value_type &kx) {
                                               return kx.first == item.first;
                                           }) -
                            l.coeff_.begin());
            applyCond(-item.second.k_, item.second.a_, cond->nodeType(),
                      lin2expr(l));
        }
    }
}

Stmt CompTransientBounds::visit(const VarDef &op) {
    if (buffers_.count(op->name_)) {
        throw InvalidProgram("Nested VarDef with the same name is not allowed");
    }
    buffers_[op->name_] = op->buffer_;
    auto ret = Mutator::visit(op);
    buffers_.erase(op->name_);
    return ret;
}

Stmt CompTransientBounds::visit(const For &op) {
    OutDatedBoundsRemover localRemover(transients_);
    localRemover(op);
    auto var = makeVar(op->iter_);
    auto hash = getHash(var);
    if (transients_.count(hash)) {
        throw InvalidProgram(
            "iterators with the same name in nested loops are not allowed");
    }
    transients_[hash] = {var, {op->begin_}, {sub1(op->end_)}};
    auto ret = Mutator::visit(op);
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

Stmt CompTransientBounds::visit(const Store &op) {
    auto ret = Mutator::visit(op);
    remover_(op);
    return ret;
}

Stmt CompTransientBounds::visit(const ReduceTo &op) {
    auto ret = Mutator::visit(op);
    remover_(op);
    return ret;
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
                old = LowerBound(LinearExpr<Rational<int>>{{}, newVal});
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
                old = UpperBound(LinearExpr<Rational<int>>{{}, newVal});
            }
            return;
        }
    }
    list.emplace_back(bound);
}

int CompUniqueBounds::getIntLower(const LowerBoundsList &list) const {
    int ret = INT_MIN;
    for (auto &&b : list) {
        if (b.lin().coeff_.empty()) {
            auto bias = b.lin().bias_;
            ret = std::max(ret, ceilDiv(bias.p_, bias.q_));
        }
    }
    return ret;
}

int CompUniqueBounds::getIntUpper(const UpperBoundsList &list) const {
    int ret = INT_MAX;
    for (auto &&b : list) {
        if (b.lin().coeff_.empty()) {
            auto bias = b.lin().bias_;
            ret = std::min(ret, floorDiv(bias.p_, bias.q_));
        }
    }
    return ret;
}

Ref<int> CompUniqueBounds::getInt(const BoundsListPair &pair) const {
    int lower = getIntLower(pair.first);
    int upper = getIntUpper(pair.second);
    return lower == upper ? Ref<int>::make(lower) : nullptr;
}

CompUniqueBounds::BoundsListPair
CompUniqueBounds::addBounds(const BoundsListPair &lhs,
                            const BoundsListPair &rhs) const {
    BoundsListPair ret;
    for (auto &&b1 : lhs.first) {
        for (auto &&b2 : rhs.first) {
            updLower(ret.first, add(b1, b2));
        }
    }
    for (auto &&b1 : lhs.second) {
        for (auto &&b2 : rhs.second) {
            updUpper(ret.second, add(b1, b2));
        }
    }
    return ret;
}

CompUniqueBounds::BoundsListPair
CompUniqueBounds::subBounds(const BoundsListPair &lhs,
                            const BoundsListPair &rhs) const {
    BoundsListPair ret;
    for (auto &&b1 : lhs.first) {
        for (auto &&b2 : rhs.second) {
            updLower(ret.first, sub(b1, b2));
        }
    }
    for (auto &&b1 : lhs.second) {
        for (auto &&b2 : rhs.first) {
            updUpper(ret.second, sub(b1, b2));
        }
    }
    return ret;
}

Expr CompUniqueBounds::visitExpr(
    const Expr &_op, const std::function<Expr(const Expr &)> &visitNode) {
    auto op = CompTransientBounds::visitExpr(_op, visitNode);
    static bool inRecur = false;
    if (!inRecur) {
        inRecur = true;
        auto tr = transient(op);
        for (auto &&_first : tr.lower_) {
            auto first = (*this)(_first);
            for (auto &&item : getLower(first)) {
                updLower(lower_[op], item);
            }
        }
        for (auto &&_second : tr.upper_) {
            auto second = (*this)(_second);
            for (auto &&item : getUpper(second)) {
                updUpper(upper_[op], item);
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
    updLower(lower_[op], LowerBound{LinearExpr<Rational<int>>{{}, op->val_}});
    updUpper(upper_[op], UpperBound{LinearExpr<Rational<int>>{{}, op->val_}});
    return op;
}

Expr CompUniqueBounds::visit(const Add &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Add);
    auto op = __op.as<AddNode>();
    setBoundsPair(op,
                  addBounds(getBoundsPair(op->lhs_), getBoundsPair(op->rhs_)));
    return op;
}

Expr CompUniqueBounds::visit(const Sub &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Sub);
    auto op = __op.as<SubNode>();
    setBoundsPair(op,
                  subBounds(getBoundsPair(op->lhs_), getBoundsPair(op->rhs_)));
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
                    updLower(lower_[op], mul(b, *k));
                }
                for (auto &&b : getUpper(e1)) {
                    updUpper(upper_[op], mul(b, *k));
                }
            } else {
                for (auto &&b : getLower(e1)) {
                    updUpper(upper_[op], mul(UpperBound{b.lin()}, *k));
                }
                for (auto &&b : getUpper(e1)) {
                    updLower(lower_[op], mul(LowerBound{b.lin()}, *k));
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
                updLower(lower_[op], floorDiv(b, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updUpper(upper_[op], floorDiv(b, *k));
            }
        } else {
            for (auto &&b : getLower(op->lhs_)) {
                updUpper(upper_[op], floorDiv(UpperBound{b.lin()}, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updLower(lower_[op], floorDiv(LowerBound{b.lin()}, *k));
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
                updLower(lower_[op], ceilDiv(b, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updUpper(upper_[op], ceilDiv(b, *k));
            }
        } else {
            for (auto &&b : getLower(op->lhs_)) {
                updUpper(upper_[op], ceilDiv(UpperBound{b.lin()}, *k));
            }
            for (auto &&b : getUpper(op->lhs_)) {
                updLower(lower_[op], ceilDiv(LowerBound{b.lin()}, *k));
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
    updLower(lower_[op], LowerBound{LinearExpr<Rational<int>>{{}, 0}});
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
    for (auto &&b : getUpper(op->lhs_)) {
        updUpper(upper_[op], b);
    }
    for (auto &&b : getUpper(op->rhs_)) {
        updUpper(upper_[op], b);
    }
    for (auto &&b1 : getLower(op->lhs_)) {
        for (auto &&b2 : getLower(op->rhs_)) {
            if (b1.lin().coeff_.empty() && b2.lin().coeff_.empty()) {
                updLower(lower_[op],
                         LinearExpr<Rational<int>>{
                             {}, std::min(b1.lin().bias_, b2.lin().bias_)});
            }
        }
    }
    updLower(lower_[op], LowerBound{op});
    updUpper(upper_[op], UpperBound{op});
    return op;
}

Expr CompUniqueBounds::visit(const Max &_op) {
    auto __op = CompTransientBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Max);
    auto op = __op.as<MaxNode>();
    if (!isInt(dtype(op))) {
        return op;
    }
    for (auto &&b : getLower(op->lhs_)) {
        updLower(lower_[op], b);
    }
    for (auto &&b : getLower(op->rhs_)) {
        updLower(lower_[op], b);
    }
    for (auto &&b1 : getUpper(op->lhs_)) {
        for (auto &&b2 : getUpper(op->rhs_)) {
            if (b1.lin().coeff_.empty() && b2.lin().coeff_.empty()) {
                updUpper(upper_[op],
                         LinearExpr<Rational<int>>{
                             {}, std::max(b1.lin().bias_, b2.lin().bias_)});
            }
        }
    }
    updLower(lower_[op], LowerBound{op});
    updUpper(upper_[op], UpperBound{op});
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
