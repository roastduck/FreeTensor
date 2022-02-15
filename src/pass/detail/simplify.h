#ifndef DETAIL_SIMPLIFY_H
#define DETAIL_SIMPLIFY_H

#include <sstream>

#include <hash.h>
#include <pass/annotate_conds.h>
#include <pass/replace_iter.h>
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

inline int countHeavyOps(const Expr &op) {
    CountHeavyOps visitor;
    visitor(op);
    return visitor.cnt();
}

} // namespace detail

template <class Unique> Expr SimplifyPass<Unique>::visitExpr(const Expr &_op) {
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
    for (auto &&lower : this->unique_.getLower(op)) {
        for (auto &&upper : this->unique_.getUpper(op)) {
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
                auto heavyOps = detail::countHeavyOps(expr);
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

template <class Unique> Expr SimplifyPass<Unique>::visit(const IntConst &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IntConst);
    auto op = __op.template as<IntConstNode>();
    constants_[op] = op->val_;
    return op;
}

template <class Unique> Expr SimplifyPass<Unique>::visit(const Add &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Add);
    auto op = __op.template as<AddNode>();

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

template <class Unique> Expr SimplifyPass<Unique>::visit(const Sub &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Sub);
    auto op = __op.template as<SubNode>();

    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return makeIntConst(constants_.at(op->lhs_) - constants_.at(op->rhs_));
    }
    if (constants_.count(op->rhs_) && constants_.at(op->rhs_) == 0) {
        return op->lhs_;
    }

    return op;
}

template <class Unique> Expr SimplifyPass<Unique>::visit(const Mul &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mul);
    auto op = __op.template as<MulNode>();

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

template <class Unique> Expr SimplifyPass<Unique>::visit(const FloorDiv &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::FloorDiv);
    auto op = __op.template as<FloorDivNode>();
    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return makeIntConst(
            floorDiv(constants_.at(op->lhs_), constants_.at(op->rhs_)));
    }
    return op;
}

template <class Unique> Expr SimplifyPass<Unique>::visit(const CeilDiv &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::CeilDiv);
    auto op = __op.template as<CeilDivNode>();
    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return makeIntConst(
            ceilDiv(constants_.at(op->lhs_), constants_.at(op->rhs_)));
    }
    return op;
}

template <class Unique>
Expr SimplifyPass<Unique>::visit(const RoundTowards0Div &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::RoundTowards0Div);
    auto op = __op.template as<RoundTowards0DivNode>();
    if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
        return makeIntConst(constants_.at(op->lhs_) / constants_.at(op->rhs_));
    }
    return op;
}

template <class Unique> Expr SimplifyPass<Unique>::visit(const Mod &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mod);
    auto op = __op.template as<ModNode>();

    if (this->unique_.getIntLower(op->rhs_) > 0 &&
        this->unique_.getIntLower(op->lhs_) >= 0 &&
        this->unique_.alwaysLT(op->lhs_, op->rhs_)) {
        return op->lhs_;
    }
    if (this->unique_.getIntUpper(op->rhs_) < 0 &&
        this->unique_.getIntUpper(op->rhs_) <= 0 &&
        this->unique_.alwaysLT(op->rhs_, op->lhs_)) {
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

template <class Unique> Expr SimplifyPass<Unique>::visit(const Remainder &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Remainder);
    auto op = __op.template as<RemainderNode>();

    if (constants_.count(op->rhs_) && constants_.count(op->lhs_)) {
        return makeIntConst(constants_.at(op->lhs_) % constants_.at(op->rhs_));
    }

    return op;
}

template <class Unique> Expr SimplifyPass<Unique>::visit(const Min &_op) {
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
            if (this->unique_.alwaysLE(l, r)) {
                all.erase(r);
            } else if (this->unique_.alwaysLE(r, l)) {
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

template <class Unique> Expr SimplifyPass<Unique>::visit(const Max &_op) {
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
            if (this->unique_.alwaysLE(l, r)) {
                all.erase(l);
            } else if (this->unique_.alwaysLE(r, l)) {
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

template <class Unique> Expr SimplifyPass<Unique>::visit(const LT &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LT);
    auto op = __op.template as<LTNode>();
    if (!isInt(this->dtype(op->lhs_)) || !isInt(this->dtype(op->rhs_))) {
        return op;
    }
    if (this->unique_.alwaysLT(op->lhs_, op->rhs_)) {
        return makeBoolConst(true);
    }
    if (this->unique_.alwaysLE(op->rhs_, op->lhs_)) {
        return makeBoolConst(false);
    }
    return op;
}

template <class Unique> Expr SimplifyPass<Unique>::visit(const LE &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LE);
    auto op = __op.template as<LENode>();
    if (!isInt(this->dtype(op->lhs_)) || !isInt(this->dtype(op->rhs_))) {
        return op;
    }
    if (this->unique_.alwaysLE(op->lhs_, op->rhs_)) {
        return makeBoolConst(true);
    }
    if (this->unique_.alwaysLT(op->rhs_, op->lhs_)) {
        return makeBoolConst(false);
    }
    return op;
}

template <class Unique> Expr SimplifyPass<Unique>::visit(const GT &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::GT);
    auto op = __op.template as<GTNode>();
    if (!isInt(this->dtype(op->lhs_)) || !isInt(this->dtype(op->rhs_))) {
        return op;
    }
    if (this->unique_.alwaysLE(op->lhs_, op->rhs_)) {
        return makeBoolConst(false);
    }
    if (this->unique_.alwaysLT(op->rhs_, op->lhs_)) {
        return makeBoolConst(true);
    }
    return op;
}

template <class Unique> Expr SimplifyPass<Unique>::visit(const GE &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::GE);
    auto op = __op.template as<GENode>();
    if (!isInt(this->dtype(op->lhs_)) || !isInt(this->dtype(op->rhs_))) {
        return op;
    }
    if (this->unique_.alwaysLT(op->lhs_, op->rhs_)) {
        return makeBoolConst(false);
    }
    if (this->unique_.alwaysLE(op->rhs_, op->lhs_)) {
        return makeBoolConst(true);
    }
    return op;
}

template <class Unique> Expr SimplifyPass<Unique>::visit(const EQ &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::EQ);
    auto op = __op.template as<EQNode>();
    if (!isInt(this->dtype(op->lhs_)) || !isInt(this->dtype(op->rhs_))) {
        return op;
    }
    if (this->unique_.alwaysLT(op->lhs_, op->rhs_)) {
        return makeBoolConst(false);
    }
    if (this->unique_.alwaysLT(op->rhs_, op->lhs_)) {
        return makeBoolConst(false);
    }
    if (this->unique_.alwaysLE(op->lhs_, op->rhs_) &&
        this->unique_.alwaysLE(op->rhs_, op->lhs_)) {
        return makeBoolConst(true);
    }
    return op;
}

template <class Unique> Expr SimplifyPass<Unique>::visit(const NE &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::NE);
    auto op = __op.template as<NENode>();
    if (!isInt(this->dtype(op->lhs_)) || !isInt(this->dtype(op->rhs_))) {
        return op;
    }
    if (this->unique_.alwaysLT(op->lhs_, op->rhs_)) {
        return makeBoolConst(true);
    }
    if (this->unique_.alwaysLT(op->rhs_, op->lhs_)) {
        return makeBoolConst(true);
    }
    if (this->unique_.alwaysLE(op->lhs_, op->rhs_) &&
        this->unique_.alwaysLE(op->rhs_, op->lhs_)) {
        return makeBoolConst(false);
    }
    return op;
}

template <class Unique> Expr SimplifyPass<Unique>::visit(const LAnd &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LAnd);
    auto op = __op.template as<LAndNode>();
    if (op->lhs_->nodeType() == ASTNodeType::BoolConst) {
        return op->lhs_.template as<BoolConstNode>()->val_
                   ? (Expr)op->rhs_
                   : makeBoolConst(false);
    }
    if (op->rhs_->nodeType() == ASTNodeType::BoolConst) {
        return op->rhs_.template as<BoolConstNode>()->val_
                   ? (Expr)op->lhs_
                   : makeBoolConst(false);
    }
    return op;
}

template <class Unique> Expr SimplifyPass<Unique>::visit(const LOr &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LOr);
    auto op = __op.template as<LOrNode>();
    if (op->lhs_->nodeType() == ASTNodeType::BoolConst) {
        return op->lhs_.template as<BoolConstNode>()->val_ ? makeBoolConst(true)
                                                           : (Expr)op->rhs_;
    }
    if (op->rhs_->nodeType() == ASTNodeType::BoolConst) {
        return op->rhs_.template as<BoolConstNode>()->val_ ? makeBoolConst(true)
                                                           : (Expr)op->lhs_;
    }
    return op;
}

template <class Unique> Expr SimplifyPass<Unique>::visit(const LNot &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LNot);
    auto op = __op.template as<LNotNode>();
    switch (op->expr_->nodeType()) {
    case ASTNodeType::BoolConst:
        return makeBoolConst(!op->expr_.template as<BoolConstNode>()->val_);
    case ASTNodeType::LT:
        return makeGE(op->expr_.template as<LTNode>()->lhs_,
                      op->expr_.template as<LTNode>()->rhs_);
    case ASTNodeType::GT:
        return makeLE(op->expr_.template as<GTNode>()->lhs_,
                      op->expr_.template as<GTNode>()->rhs_);
    case ASTNodeType::LE:
        return makeGT(op->expr_.template as<LENode>()->lhs_,
                      op->expr_.template as<LENode>()->rhs_);
    case ASTNodeType::GE:
        return makeLT(op->expr_.template as<GENode>()->lhs_,
                      op->expr_.template as<GENode>()->rhs_);
    case ASTNodeType::EQ:
        return makeNE(op->expr_.template as<EQNode>()->lhs_,
                      op->expr_.template as<EQNode>()->rhs_);
    case ASTNodeType::NE:
        return makeEQ(op->expr_.template as<NENode>()->lhs_,
                      op->expr_.template as<NENode>()->rhs_);
    case ASTNodeType::LAnd:
        return makeLOr(makeLNot(op->expr_.template as<LAndNode>()->lhs_),
                       makeLNot(op->expr_.template as<LAndNode>()->rhs_));
    case ASTNodeType::LOr:
        return makeLAnd(makeLNot(op->expr_.template as<LOrNode>()->lhs_),
                        makeLNot(op->expr_.template as<LOrNode>()->rhs_));
    case ASTNodeType::LNot:
        return op->expr_.template as<LNotNode>()->expr_;
    default:;
    }
    return op;
}

template <class Unique> Expr SimplifyPass<Unique>::visit(const IfExpr &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IfExpr);
    auto op = __op.template as<IfExprNode>();
    if (op->cond_->nodeType() == ASTNodeType::BoolConst) {
        if (op->cond_.template as<BoolConstNode>()->val_) {
            return op->thenCase_;
        } else {
            return op->elseCase_;
        }
    }
    return op;
}

template <class Unique> Stmt SimplifyPass<Unique>::visit(const ReduceTo &_op) {
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

template <class Unique> Stmt SimplifyPass<Unique>::visit(const VarDef &_op) {
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
        if (this->unique_.getIntLower(makeSub(op->sizeLim_, size)) >= 0) {
            op->sizeLim_ = nullptr;
        }
    }

    return op;
}

template <class Unique> Stmt SimplifyPass<Unique>::visit(const For &_op) {
    if (varScope_.count(_op->iter_)) {
        throw InvalidProgram(
            "iterators with the same name in nested loops are not allowed");
    }

    varScope_[_op->iter_] = curScope_++;
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.template as<ForNode>();
    varScope_.erase(_op->iter_), curScope_--;

    if (auto intLen_ = this->unique_.getInt(op->len_); intLen_.isValid()) {
        auto intLen = *intLen_;
        if (intLen == 1) {
            auto body = ReplaceIter(_op->iter_, op->begin_)(_op->body_);
            return (*this)(body);
        }
        if (intLen <= 0) {
            return makeStmtSeq("", {});
        }
    }
    if (this->unique_.getIntUpper(op->len_) == 1) {
        auto body = ReplaceIter(_op->iter_, op->begin_)(_op->body_);
        body = (*this)(body);
        return makeIf("", makeEQ(op->len_, makeIntConst(1)), body);
    }

    if (detail::isEmptyStmt(op->body_)) {
        return makeStmtSeq("", {});
    }
    return op;
}

template <class Unique> Stmt SimplifyPass<Unique>::visit(const If &_op) {
    // Simplify the condition first to determine a possible dead branch, so we
    // can avoid recurse into the dead branch. This allows assertion false in
    // the dead branch
    auto cond = (*this)(_op->cond_);
    if (cond->nodeType() == ASTNodeType::BoolConst) {
        if (cond.template as<BoolConstNode>()->val_) {
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
            .template as<IfNode>());
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.template as<IfNode>();
    bool emptyThen = detail::isEmptyStmt(op->thenCase_);
    bool emptyElse = detail::isEmptyStmt(op->elseCase_);
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

template <class Unique> Stmt SimplifyPass<Unique>::visit(const Assert &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Assert);
    auto op = __op.template as<AssertNode>();
    if (op->cond_->nodeType() == ASTNodeType::BoolConst) {
        if (op->cond_.template as<BoolConstNode>()->val_) {
            return op->body_;
        } else {
            // Print the unchanged _op
            throw AssertAlwaysFalse("Assertion always false: " + toString(_op));
        }
    }
    return op;
}

template <class Simplifier>
std::tuple<Stmt, typename CompUniqueBounds::LowerBoundsMap,
           typename CompUniqueBounds::UpperBoundsMap>
simplifyAndGetBounds(const Stmt &_op) {
    auto op = _op;

    for (int i = 0;; i++) {
        op = annotateConds(op);

        Simplifier mutator;
        auto newOp = mutator(op);

        if (HashComparator()(newOp, op) || i > 100) {
            if (i > 100) {
                WARNING("SimplifyPass iterates over 100 rounds. Maybe there is "
                        "a bug");
            }
            return {newOp, mutator.uniqueBounds().lower(),
                    mutator.uniqueBounds().upper()};
        }
        op = newOp;
    }
}

} // namespace ir

#endif // DETAIL_SIMPLIFY_H
