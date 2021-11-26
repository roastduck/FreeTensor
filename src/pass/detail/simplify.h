#ifndef DETAIL_SIMPLIFY_H
#define DETAIL_SIMPLIFY_H

#include <sstream>

#include <pass/simplify.h>

namespace ir {

namespace detail {

inline bool isEmptyStmt(const Stmt &op) {
    if (!op.isValid()) { // In case If->elseCase_ == nullptr
        return true;
    }
    if (op->nodeType() == ASTNodeType::StmtSeq &&
        op.as<StmtSeqNode>()->stmts_.empty()) {
        return true;
    }
    return false;
}

} // namespace detail

template <class BaseClass>
Expr SimplifyPass<BaseClass>::visitExpr(
    const Expr &_op, const std::function<Expr(const Expr &)> &visitNode) {
    auto op = BaseClass::visitExpr(_op, visitNode);

    // To avoid divergence
    if (this->getHash(op) != this->getHash(_op)) {
        // E.g.
        // (1) a[0 - 0] -> a[0]
        // (2) (1 + 1) * a[0] -> 2 * a[0 - 0], because of the old bound
        return op;
    }

    Expr best = nullptr;
    auto bestScope = -1;
    for (auto &&lower : this->getLower(op)) {
        for (auto &&upper : this->getUpper(op)) {
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
                if (!best.isValid() || scope < bestScope) {
                    best = expr, bestScope = scope;
                }
                break;
            }
        }
    }
    if (best.isValid() && this->getHash(best) != this->getHash(op)) {
        return markMutated(best);
    }
    return op;
}

template <class BaseClass>
Expr SimplifyPass<BaseClass>::visit(const IntConst &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IntConst);
    auto op = __op.template as<IntConstNode>();
    constants_[op] = op->val_;
    return op;
}

template <class BaseClass> Expr SimplifyPass<BaseClass>::visit(const Add &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Add);
    auto op = __op.template as<AddNode>();

    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return markMutated(
            makeIntConst(constants_.at(op->lhs_) + constants_.at(op->rhs_)));
    }
    if (constants_.count(op->lhs_) && constants_.at(op->lhs_) == 0) {
        return markMutated(op->rhs_);
    }
    if (constants_.count(op->rhs_) && constants_.at(op->rhs_) == 0) {
        return markMutated(op->lhs_);
    }

    return op;
}

template <class BaseClass> Expr SimplifyPass<BaseClass>::visit(const Sub &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Sub);
    auto op = __op.template as<SubNode>();

    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return markMutated(
            makeIntConst(constants_.at(op->lhs_) - constants_.at(op->rhs_)));
    }
    if (constants_.count(op->rhs_) && constants_.at(op->rhs_) == 0) {
        return markMutated(op->lhs_);
    }

    return op;
}

template <class BaseClass> Expr SimplifyPass<BaseClass>::visit(const Mul &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mul);
    auto op = __op.template as<MulNode>();

    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return markMutated(
            makeIntConst(constants_.at(op->lhs_) * constants_.at(op->rhs_)));
    }
    if (constants_.count(op->lhs_) && constants_.at(op->lhs_) == 1) {
        return markMutated(op->rhs_);
    }
    if (constants_.count(op->rhs_) && constants_.at(op->rhs_) == 1) {
        return markMutated(op->lhs_);
    }
    if (constants_.count(op->lhs_) && constants_.at(op->lhs_) == 0) {
        return markMutated(makeIntConst(0));
    }
    if (constants_.count(op->rhs_) && constants_.at(op->rhs_) == 0) {
        return markMutated(makeIntConst(0));
    }

    return op;
}

template <class BaseClass> Expr SimplifyPass<BaseClass>::visit(const Var &op) {
    if (replace_.count(op->name_)) {
        return (*this)(replace_.at(op->name_));
    }
    return BaseClass::visit(op);
}

template <class BaseClass>
Expr SimplifyPass<BaseClass>::visit(const FloorDiv &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::FloorDiv);
    auto op = __op.template as<FloorDivNode>();
    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return markMutated(makeIntConst(
            floorDiv(constants_.at(op->lhs_), constants_.at(op->rhs_))));
    }
    return op;
}

template <class BaseClass>
Expr SimplifyPass<BaseClass>::visit(const CeilDiv &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::CeilDiv);
    auto op = __op.template as<CeilDivNode>();
    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return markMutated(makeIntConst(
            ceilDiv(constants_.at(op->lhs_), constants_.at(op->rhs_))));
    }
    return op;
}

template <class BaseClass>
Expr SimplifyPass<BaseClass>::visit(const RoundTowards0Div &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::RoundTowards0Div);
    auto op = __op.template as<RoundTowards0DivNode>();
    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return markMutated(
            makeIntConst(constants_.at(op->lhs_) / constants_.at(op->rhs_)));
    }
    return op;
}

template <class BaseClass> Expr SimplifyPass<BaseClass>::visit(const Mod &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mod);
    auto op = __op.template as<ModNode>();

    if (this->getIntLower(op->rhs_) > 0 && this->getIntLower(op->lhs_) >= 0 &&
        this->alwaysLT(op->lhs_, op->rhs_)) {
        return markMutated(op->lhs_);
    }
    if (this->getIntUpper(op->rhs_) < 0 && this->getIntUpper(op->rhs_) <= 0 &&
        this->alwaysLT(op->rhs_, op->lhs_)) {
        return markMutated(op->lhs_);
    }

    if (constants_.count(op->rhs_)) {
        auto k = constants_.at(op->rhs_);

        if (constants_.count(op->lhs_)) {
            return markMutated(makeIntConst(mod(constants_.at(op->lhs_), k)));
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
            return markMutated(makeMod(newLhs, op->rhs_));
        }
    }

    return op;
}

template <class BaseClass>
Expr SimplifyPass<BaseClass>::visit(const Remainder &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Remainder);
    auto op = __op.template as<RemainderNode>();

    if (constants_.count(op->rhs_) && constants_.count(op->lhs_)) {
        return markMutated(
            makeIntConst(constants_.at(op->lhs_) % constants_.at(op->rhs_)));
    }

    return op;
}

template <class BaseClass> Expr SimplifyPass<BaseClass>::visit(const Min &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Min);
    auto op = __op.template as<MinNode>();

    // Followed by rules only for integers
    if (!isInt(this->dtype(op))) {
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
            if (this->alwaysLE(l, r)) {
                all.erase(r);
            } else if (this->alwaysLE(r, l)) {
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

template <class BaseClass> Expr SimplifyPass<BaseClass>::visit(const Max &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Max);
    auto op = __op.template as<MaxNode>();

    // Followed by rules only for integers
    if (!isInt(this->dtype(op))) {
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
            if (this->alwaysLE(l, r)) {
                all.erase(l);
            } else if (this->alwaysLE(r, l)) {
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

template <class BaseClass> Expr SimplifyPass<BaseClass>::visit(const LT &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LT);
    auto op = __op.template as<LTNode>();
    if (!isInt(this->dtype(op->lhs_)) || !isInt(this->dtype(op->rhs_))) {
        return op;
    }
    if (this->alwaysLT(op->lhs_, op->rhs_)) {
        return markMutated(makeBoolConst(true));
    }
    if (this->alwaysLE(op->rhs_, op->lhs_)) {
        return markMutated(makeBoolConst(false));
    }
    return op;
}

template <class BaseClass> Expr SimplifyPass<BaseClass>::visit(const LE &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LE);
    auto op = __op.template as<LENode>();
    if (!isInt(this->dtype(op->lhs_)) || !isInt(this->dtype(op->rhs_))) {
        return op;
    }
    if (this->alwaysLE(op->lhs_, op->rhs_)) {
        return markMutated(makeBoolConst(true));
    }
    if (this->alwaysLT(op->rhs_, op->lhs_)) {
        return markMutated(makeBoolConst(false));
    }
    return op;
}

template <class BaseClass> Expr SimplifyPass<BaseClass>::visit(const GT &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::GT);
    auto op = __op.template as<GTNode>();
    if (!isInt(this->dtype(op->lhs_)) || !isInt(this->dtype(op->rhs_))) {
        return op;
    }
    if (this->alwaysLE(op->lhs_, op->rhs_)) {
        return markMutated(makeBoolConst(false));
    }
    if (this->alwaysLT(op->rhs_, op->lhs_)) {
        return markMutated(makeBoolConst(true));
    }
    return op;
}

template <class BaseClass> Expr SimplifyPass<BaseClass>::visit(const GE &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::GE);
    auto op = __op.template as<GENode>();
    if (!isInt(this->dtype(op->lhs_)) || !isInt(this->dtype(op->rhs_))) {
        return op;
    }
    if (this->alwaysLT(op->lhs_, op->rhs_)) {
        return markMutated(makeBoolConst(false));
    }
    if (this->alwaysLE(op->rhs_, op->lhs_)) {
        return markMutated(makeBoolConst(true));
    }
    return op;
}

template <class BaseClass> Expr SimplifyPass<BaseClass>::visit(const EQ &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::EQ);
    auto op = __op.template as<EQNode>();
    if (!isInt(this->dtype(op->lhs_)) || !isInt(this->dtype(op->rhs_))) {
        return op;
    }
    if (this->alwaysLT(op->lhs_, op->rhs_)) {
        return markMutated(makeBoolConst(false));
    }
    if (this->alwaysLT(op->rhs_, op->lhs_)) {
        return markMutated(makeBoolConst(false));
    }
    if (this->alwaysLE(op->lhs_, op->rhs_) &&
        this->alwaysLE(op->rhs_, op->lhs_)) {
        return markMutated(makeBoolConst(true));
    }
    return op;
}

template <class BaseClass> Expr SimplifyPass<BaseClass>::visit(const NE &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::NE);
    auto op = __op.template as<NENode>();
    if (!isInt(this->dtype(op->lhs_)) || !isInt(this->dtype(op->rhs_))) {
        return op;
    }
    if (this->alwaysLT(op->lhs_, op->rhs_)) {
        return markMutated(makeBoolConst(true));
    }
    if (this->alwaysLT(op->rhs_, op->lhs_)) {
        return markMutated(makeBoolConst(true));
    }
    if (this->alwaysLE(op->lhs_, op->rhs_) &&
        this->alwaysLE(op->rhs_, op->lhs_)) {
        return markMutated(makeBoolConst(false));
    }
    return op;
}

template <class BaseClass>
Expr SimplifyPass<BaseClass>::visit(const LAnd &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LAnd);
    auto op = __op.template as<LAndNode>();
    if (op->lhs_->nodeType() == ASTNodeType::BoolConst) {
        return markMutated(op->lhs_.template as<BoolConstNode>()->val_
                               ? (Expr)op->rhs_
                               : makeBoolConst(false));
    }
    if (op->rhs_->nodeType() == ASTNodeType::BoolConst) {
        return markMutated(op->rhs_.template as<BoolConstNode>()->val_
                               ? (Expr)op->lhs_
                               : makeBoolConst(false));
    }
    return op;
}

template <class BaseClass> Expr SimplifyPass<BaseClass>::visit(const LOr &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LOr);
    auto op = __op.template as<LOrNode>();
    if (op->lhs_->nodeType() == ASTNodeType::BoolConst) {
        return markMutated(op->lhs_.template as<BoolConstNode>()->val_
                               ? makeBoolConst(true)
                               : (Expr)op->rhs_);
    }
    if (op->rhs_->nodeType() == ASTNodeType::BoolConst) {
        return markMutated(op->rhs_.template as<BoolConstNode>()->val_
                               ? makeBoolConst(true)
                               : (Expr)op->lhs_);
    }
    return op;
}

template <class BaseClass>
Expr SimplifyPass<BaseClass>::visit(const LNot &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LNot);
    auto op = __op.template as<LNotNode>();
    switch (op->expr_->nodeType()) {
    case ASTNodeType::BoolConst:
        return markMutated(
            makeBoolConst(!op->expr_.template as<BoolConstNode>()->val_));
    case ASTNodeType::LT:
        return markMutated(makeGE(op->expr_.template as<LTNode>()->lhs_,
                                  op->expr_.template as<LTNode>()->rhs_));
    case ASTNodeType::GT:
        return markMutated(makeLE(op->expr_.template as<GTNode>()->lhs_,
                                  op->expr_.template as<GTNode>()->rhs_));
    case ASTNodeType::LE:
        return markMutated(makeGT(op->expr_.template as<LENode>()->lhs_,
                                  op->expr_.template as<LENode>()->rhs_));
    case ASTNodeType::GE:
        return markMutated(makeLT(op->expr_.template as<GENode>()->lhs_,
                                  op->expr_.template as<GENode>()->rhs_));
    case ASTNodeType::EQ:
        return markMutated(makeNE(op->expr_.template as<EQNode>()->lhs_,
                                  op->expr_.template as<EQNode>()->rhs_));
    case ASTNodeType::NE:
        return markMutated(makeEQ(op->expr_.template as<NENode>()->lhs_,
                                  op->expr_.template as<NENode>()->rhs_));
    case ASTNodeType::LAnd:
        return markMutated(
            makeLOr(makeLNot(op->expr_.template as<LAndNode>()->lhs_),
                    makeLNot(op->expr_.template as<LAndNode>()->rhs_)));
    case ASTNodeType::LOr:
        return markMutated(
            makeLAnd(makeLNot(op->expr_.template as<LOrNode>()->lhs_),
                     makeLNot(op->expr_.template as<LOrNode>()->rhs_)));
    case ASTNodeType::LNot:
        return markMutated(op->expr_.template as<LNotNode>()->expr_);
    default:;
    }
    return op;
}

template <class BaseClass>
Expr SimplifyPass<BaseClass>::visit(const IfExpr &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IfExpr);
    auto op = __op.template as<IfExprNode>();
    if (op->cond_->nodeType() == ASTNodeType::BoolConst) {
        if (op->cond_.template as<BoolConstNode>()->val_) {
            return markMutated(op->thenCase_);
        } else {
            return markMutated(op->elseCase_);
        }
    }
    return op;
}

template <class BaseClass>
Stmt SimplifyPass<BaseClass>::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.template as<ReduceToNode>();
    switch (op->op_) {
    case ReduceOp::Add:
        if (op->expr_->nodeType() == ASTNodeType::IntConst &&
            op->expr_.template as<IntConstNode>()->val_ == 0) {
            return makeStmtSeq("", {});
        }
        if (op->expr_->nodeType() == ASTNodeType::FloatConst &&
            op->expr_.template as<FloatConstNode>()->val_ == 0) {
            return makeStmtSeq("", {});
        }
        break;
    case ReduceOp::Mul:
        if (op->expr_->nodeType() == ASTNodeType::IntConst &&
            op->expr_.template as<IntConstNode>()->val_ == 1) {
            return makeStmtSeq("", {});
        }
        if (op->expr_->nodeType() == ASTNodeType::FloatConst &&
            op->expr_.template as<FloatConstNode>()->val_ == 1) {
            return makeStmtSeq("", {});
        }
        break;
    default:; // do nothing
    }
    return op;
}

template <class BaseClass>
Stmt SimplifyPass<BaseClass>::visit(const VarDef &_op) {
    if (varScope_.count(_op->name_)) {
        throw InvalidProgram(
            "Conflict var name: " + _op->name_ +
            ". Nested vars with the same name are not allowed");
    }
    varScope_[_op->name_] = curScope_++;
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.template as<VarDefNode>();
    varScope_.erase(_op->name_), curScope_--;

    if (detail::isEmptyStmt(op->body_)) {
        return makeStmtSeq("", {});
    }

    if (op->sizeLim_.isValid()) {
        Expr size = makeIntConst(1);
        for (auto &&dim : op->buffer_->tensor().shape()) {
            size = makeMul(size, dim);
        }
        if (this->getIntLower((*this)(makeSub(op->sizeLim_, size))) >= 0) {
            op->sizeLim_ = nullptr;
        }
    }

    return op;
}

template <class BaseClass> Stmt SimplifyPass<BaseClass>::visit(const For &_op) {
    if (varScope_.count(_op->iter_)) {
        throw InvalidProgram(
            "iterators with the same name in nested loops are not allowed");
    }

    varScope_[_op->iter_] = curScope_++;
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.template as<ForNode>();
    varScope_.erase(_op->iter_), curScope_--;

    if (auto intLen_ = this->getInt(op->len_); intLen_.isValid()) {
        auto intLen = *intLen_;
        if (intLen == 1) {
            ASSERT(!replace_.count(_op->iter_));
            replace_[_op->iter_] = op->begin_;
            auto body = (*this)(_op->body_);
            replace_.erase(_op->iter_);
            return body;
        }
        if (intLen <= 0) {
            return makeStmtSeq("", {});
        }
    }
    if (this->getIntUpper(op->len_) == 1) {
        ASSERT(!replace_.count(_op->iter_));
        replace_[_op->iter_] = op->begin_;
        auto body = (*this)(_op->body_);
        replace_.erase(_op->iter_);
        return markMutated(makeIf("", makeEQ(op->len_, makeIntConst(1)), body));
    }

    if (detail::isEmptyStmt(op->body_)) {
        return makeStmtSeq("", {});
    }
    return op;
}

template <class BaseClass> Stmt SimplifyPass<BaseClass>::visit(const If &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.template as<IfNode>();
    bool emptyThen = detail::isEmptyStmt(op->thenCase_);
    bool emptyElse = detail::isEmptyStmt(op->elseCase_);
    if (emptyThen && emptyElse) {
        return makeStmtSeq("", {});
    }
    if (op->cond_->nodeType() == ASTNodeType::BoolConst) {
        if (op->cond_.template as<BoolConstNode>()->val_) {
            return markMutated(op->thenCase_);
        } else {
            if (op->elseCase_.isValid()) {
                return markMutated(op->elseCase_);
            } else {
                return markMutated(makeStmtSeq("", {}));
            }
        }
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

template <class BaseClass>
Stmt SimplifyPass<BaseClass>::visit(const Assert &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Assert);
    auto op = __op.template as<AssertNode>();
    if (op->cond_->nodeType() == ASTNodeType::BoolConst) {
        if (op->cond_.template as<BoolConstNode>()->val_) {
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

template <class Simplifier>
std::tuple<Stmt, typename Simplifier::LowerBoundsMap,
           typename Simplifier::UpperBoundsMap>
simplifyAndGetBounds(const Stmt &_op) {
    auto op = _op;

    for (int i = 0;; i++) {
        Simplifier mutator;
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

#endif // DETAIL_SIMPLIFY_H
