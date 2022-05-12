#include <algorithm>
#include <unordered_set>

#include <except.h>
#include <math/utils.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/replace_iter.h>
#include <pass/simplify.h>

namespace freetensor {

static bool isEmptyStmt(const Stmt &op) {
    if (!op.isValid()) { // In case If->elseCase_ == nullptr
        return true;
    }
    if (op->nodeType() == ASTNodeType::StmtSeq) {
        for (auto &&stmt : op.as<StmtSeqNode>()->stmts_) {
            if (!isEmptyStmt(stmt)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

class CountHeavyOps : public Visitor {
    int cnt_ = 0;

  public:
    int cnt() const { return cnt_; }

  protected:
    void visitExpr(const Expr &op) {
        Visitor::visitExpr(op);
        if (!op->isConst() && op->nodeType() != ASTNodeType::Add &&
            op->nodeType() != ASTNodeType::Sub &&
            op->nodeType() != ASTNodeType::Mul) {
            cnt_++;
        }
    }
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
            // Check upper <= lower ==> equal
            // Here we use the less precise alwaysLE instead of analyzing bounds
            // of `upper - lower`, in order to avoid infinite recursion
            if (freetensor::alwaysLE(upper, lower)) {
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

    return op;
}

Expr SimplifyPass::visit(const Sub &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::Sub);
    auto op = __op.as<SubNode>();

    if (equals(op->rhs_, 0)) {
        return op->lhs_;
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

    return op;
}

Expr SimplifyPass::visit(const FloorDiv &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::FloorDiv);
    auto op = __op.as<FloorDivNode>();
    if (equals(op->rhs_, 1)) {
        return op->lhs_;
    }
    if (equals(op->rhs_, -1)) {
        return makeMul(makeIntConst(-1), op->lhs_);
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
    if (equals(op->rhs_, 1)) {
        return op->lhs_;
    }
    if (equals(op->rhs_, -1)) {
        return makeMul(makeIntConst(-1), op->lhs_);
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
    if (equals(op->rhs_, 1)) {
        return op->lhs_;
    }
    if (equals(op->rhs_, -1)) {
        return makeMul(makeIntConst(-1), op->lhs_);
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

    if (op->rhs_->nodeType() == ASTNodeType::IntConst) {
        auto k = op->rhs_.as<IntConstNode>()->val_;

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

Expr SimplifyPass::visit(const LT &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
    ASSERT(__op->nodeType() == ASTNodeType::LT);
    auto op = __op.as<LTNode>();
    if (!isInt(dtype(op->lhs_)) || !isInt(dtype(op->rhs_))) {
        return op;
    }
    auto diff = makeSub(op->lhs_, op->rhs_);
    if (unique_.getIntUpper(diff) < 0) {
        return makeBoolConst(true);
    }
    if (unique_.getIntLower(diff) >= 0) {
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
    if (!isInt(dtype(op->lhs_)) || !isInt(dtype(op->rhs_))) {
        return op;
    }
    auto diff = makeSub(op->lhs_, op->rhs_);
    if (unique_.getIntUpper(diff) <= 0) {
        return makeBoolConst(true);
    }
    if (unique_.getIntLower(diff) > 0) {
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
    if (!isInt(dtype(op->lhs_)) || !isInt(dtype(op->rhs_))) {
        return op;
    }
    auto diff = makeSub(op->lhs_, op->rhs_);
    if (unique_.getIntLower(diff) > 0) {
        return makeBoolConst(true);
    }
    if (unique_.getIntUpper(diff) <= 0) {
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
    if (!isInt(dtype(op->lhs_)) || !isInt(dtype(op->rhs_))) {
        return op;
    }
    auto diff = makeSub(op->lhs_, op->rhs_);
    if (unique_.getIntLower(diff) >= 0) {
        return makeBoolConst(true);
    }
    if (unique_.getIntUpper(diff) < 0) {
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
    if (!isInt(dtype(op->lhs_)) || !isInt(dtype(op->rhs_))) {
        return op;
    }
    auto diff = makeSub(op->lhs_, op->rhs_);
    if (unique_.getIntLower(diff) > 0) {
        return makeBoolConst(false);
    }
    if (unique_.getIntUpper(diff) < 0) {
        return makeBoolConst(false);
    }
    if (auto &&c = unique_.getInt(diff); c.isValid() && *c == 0) {
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
    if (!isInt(dtype(op->lhs_)) || !isInt(dtype(op->rhs_))) {
        return op;
    }
    auto diff = makeSub(op->lhs_, op->rhs_);
    if (unique_.getIntLower(diff) > 0) {
        return makeBoolConst(true);
    }
    if (unique_.getIntUpper(diff) < 0) {
        return makeBoolConst(true);
    }
    if (auto &&c = unique_.getInt(diff); c.isValid() && *c == 0) {
        return makeBoolConst(false);
    }
    return op;
}

Expr SimplifyPass::visit(const LAnd &_op) {
    auto __op = BaseClass::visit(_op);
    if (__op->isConst()) {
        return __op;
    }
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
    if (__op->isConst()) {
        return __op;
    }
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
    if (__op->isConst()) {
        return __op;
    }
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
    case ReduceOp::LAnd:
        if (op->expr_->nodeType() == ASTNodeType::BoolConst &&
            op->expr_.as<BoolConstNode>()->val_ == true) {
            return makeStmtSeq("", {});
        }
        break;
    case ReduceOp::LOr:
        if (op->expr_->nodeType() == ASTNodeType::BoolConst &&
            op->expr_.as<BoolConstNode>()->val_ == false) {
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
    return flattenStmtSeq(simplifyImpl<BuiltinSimplify>(op));
}

Stmt simplify(const Stmt &op) { return builtinSimplify(op); }

} // namespace freetensor
