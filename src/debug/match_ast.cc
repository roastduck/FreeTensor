#include <algorithm>

#include <itertools.hpp>

#include <debug/match_ast.h>
#include <pass/flatten_stmt_seq.h>

namespace ir {

bool MatchVisitor::matchName(const std::string &thisName,
                             const std::string &otherName) {
    if (!nameMap_.count(thisName)) {
        if (nameMapImage_.count(otherName)) {
            return false;
        }
        nameMap_[thisName] = otherName;
        nameMapImage_.insert(otherName);
        return true;
    }
    return nameMap_.at(thisName) == otherName;
}

void MatchVisitor::clearName(const std::string &thisName) {
    ASSERT(nameMap_.count(thisName));
    nameMapImage_.erase(nameMap_.at(thisName));
    nameMap_.erase(thisName);
}

#define CHECK(expr)                                                            \
    if (!(expr)) {                                                             \
        isMatched_ = false;                                                    \
        return;                                                                \
    }

#define TRY_RECURSE(lexpr, rexpr)                                              \
    {                                                                          \
        auto oldInstance = instance_;                                          \
        instance_ = rexpr;                                                     \
        (*this)(lexpr);                                                        \
        instance_ = oldInstance;                                               \
    }

#define RECURSE(lexpr, rexpr)                                                  \
    {                                                                          \
        TRY_RECURSE(lexpr, rexpr)                                              \
        if (!isMatched_) {                                                     \
            return;                                                            \
        }                                                                      \
    }

void MatchVisitor::visit(const StmtSeq &op) {
    CHECK(instance_->nodeType() == ASTNodeType::StmtSeq);
    auto instance = instance_.as<StmtSeqNode>();
    CHECK(op->stmts_.size() == instance->stmts_.size());
    for (auto &&[oStmt, iStmt] : iter::zip(op->stmts_, instance->stmts_)) {
        RECURSE(oStmt, iStmt);
    }
}

void MatchVisitor::visit(const VarDef &op) {
    CHECK(instance_->nodeType() == ASTNodeType::VarDef);
    auto instance = instance_.as<VarDefNode>();
    CHECK(matchName(op->name_, instance->name_));
    CHECK(op->buffer_->atype() == instance->buffer_->atype());
    CHECK(op->buffer_->tensor().dtype() == instance->buffer_->tensor().dtype());
    auto &&lshape = op->buffer_->tensor().shape();
    auto &&rshape = instance->buffer_->tensor().shape();
    CHECK(lshape.size() == rshape.size());
    for (auto &&[ldim, rdim] : iter::zip(lshape, rshape)) {
        RECURSE(ldim, rdim);
    }
    RECURSE(op->body_, instance->body_);
    clearName(op->name_);
}

void MatchVisitor::visit(const Var &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Var);
    auto instance = instance_.as<VarNode>();
    CHECK(matchName(op->name_, instance->name_));
}

void MatchVisitor::visit(const Store &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Store);
    auto instance = instance_.as<StoreNode>();
    CHECK(matchName(op->var_, instance->var_));
    for (auto &&[oIdx, iIdx] : iter::zip(op->indices_, instance->indices_)) {
        RECURSE(oIdx, iIdx);
    }
    RECURSE(op->expr_, instance->expr_);
}

void MatchVisitor::visit(const Load &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Load);
    auto instance = instance_.as<LoadNode>();
    CHECK(matchName(op->var_, instance->var_));
    for (auto &&[oIdx, iIdx] : iter::zip(op->indices_, instance->indices_)) {
        RECURSE(oIdx, iIdx);
    }
}

void MatchVisitor::visit(const ReduceTo &op) {
    CHECK(instance_->nodeType() == ASTNodeType::ReduceTo);
    auto instance = instance_.as<ReduceToNode>();
    CHECK(matchName(op->var_, instance->var_));
    for (auto &&[oIdx, iIdx] : iter::zip(op->indices_, instance->indices_)) {
        RECURSE(oIdx, iIdx);
    }
    CHECK(op->op_ == instance->op_);
    RECURSE(op->expr_, instance->expr_);
}

void MatchVisitor::visit(const IntConst &op) {
    CHECK(instance_->nodeType() == ASTNodeType::IntConst);
    auto instance = instance_.as<IntConstNode>();
    CHECK(op->val_ == instance->val_);
}

void MatchVisitor::visit(const FloatConst &op) {
    CHECK(instance_->nodeType() == ASTNodeType::FloatConst);
    auto instance = instance_.as<FloatConstNode>();
    CHECK(op->val_ == instance->val_);
}

void MatchVisitor::visit(const BoolConst &op) {
    CHECK(instance_->nodeType() == ASTNodeType::BoolConst);
    auto instance = instance_.as<BoolConstNode>();
    CHECK(op->val_ == instance->val_);
}

void MatchVisitor::visit(const Add &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Add);
    auto instance = instance_.as<AddNode>();

    std::vector<Expr> thisOperands, instanceOperands;
    std::function<void(const Expr &, std::vector<Expr> &)> recur =
        [&recur](const Expr op, std::vector<Expr> &operands) {
            if (op->nodeType() == ASTNodeType::Add) {
                recur(op.as<AddNode>()->lhs_, operands);
                recur(op.as<AddNode>()->rhs_, operands);
            } else {
                operands.emplace_back(op);
            }
        };
    recur(op, thisOperands);
    recur(instance, instanceOperands);
    // FIXME: If the expression contains `AnyExpr`, this assertion may lead to
    // false mismatch
    CHECK(thisOperands.size() == instanceOperands.size());

    std::sort(thisOperands.begin(), thisOperands.end());
    do {
        isMatched_ = true;
        for (auto &&[oOp, iOp] : iter::zip(thisOperands, instanceOperands)) {
            TRY_RECURSE(oOp, iOp);
            if (!isMatched_) {
                goto fail;
            }
        }
        return;
    fail:;
    } while (std::next_permutation(thisOperands.begin(), thisOperands.end()));
}

void MatchVisitor::visit(const Sub &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Sub);
    auto instance = instance_.as<SubNode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const Mul &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Mul);
    auto instance = instance_.as<MulNode>();

    std::vector<Expr> thisOperands, instanceOperands;
    std::function<void(const Expr &, std::vector<Expr> &)> recur =
        [&recur](const Expr op, std::vector<Expr> &operands) {
            if (op->nodeType() == ASTNodeType::Mul) {
                recur(op.as<MulNode>()->lhs_, operands);
                recur(op.as<MulNode>()->rhs_, operands);
            } else {
                operands.emplace_back(op);
            }
        };
    recur(op, thisOperands);
    recur(instance, instanceOperands);
    CHECK(thisOperands.size() == instanceOperands.size());

    std::sort(thisOperands.begin(), thisOperands.end());
    do {
        isMatched_ = true;
        for (auto &&[oOp, iOp] : iter::zip(thisOperands, instanceOperands)) {
            TRY_RECURSE(oOp, iOp);
            if (!isMatched_) {
                goto fail;
            }
        }
        return;
    fail:;
    } while (std::next_permutation(thisOperands.begin(), thisOperands.end()));
}

void MatchVisitor::visit(const RealDiv &op) {
    CHECK(instance_->nodeType() == ASTNodeType::RealDiv);
    auto instance = instance_.as<RealDivNode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const FloorDiv &op) {
    CHECK(instance_->nodeType() == ASTNodeType::FloorDiv);
    auto instance = instance_.as<FloorDivNode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const CeilDiv &op) {
    CHECK(instance_->nodeType() == ASTNodeType::CeilDiv);
    auto instance = instance_.as<CeilDivNode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const RoundTowards0Div &op) {
    CHECK(instance_->nodeType() == ASTNodeType::RoundTowards0Div);
    auto instance = instance_.as<RoundTowards0DivNode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const Mod &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Mod);
    auto instance = instance_.as<ModNode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const Remainder &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Remainder);
    auto instance = instance_.as<RemainderNode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const Min &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Min);
    auto instance = instance_.as<MinNode>();
    TRY_RECURSE(op->lhs_, instance->lhs_);
    TRY_RECURSE(op->rhs_, instance->rhs_);
    if (!isMatched_) {
        isMatched_ = true;
        RECURSE(op->lhs_, instance->rhs_);
        RECURSE(op->rhs_, instance->lhs_);
    }
}

void MatchVisitor::visit(const Max &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Max);
    auto instance = instance_.as<MaxNode>();
    TRY_RECURSE(op->lhs_, instance->lhs_);
    TRY_RECURSE(op->rhs_, instance->rhs_);
    if (!isMatched_) {
        isMatched_ = true;
        RECURSE(op->lhs_, instance->rhs_);
        RECURSE(op->rhs_, instance->lhs_);
    }
}

void MatchVisitor::visit(const LT &op) {
    CHECK(instance_->nodeType() == ASTNodeType::LT);
    auto instance = instance_.as<LTNode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const LE &op) {
    CHECK(instance_->nodeType() == ASTNodeType::LE);
    auto instance = instance_.as<LENode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const GT &op) {
    CHECK(instance_->nodeType() == ASTNodeType::GT);
    auto instance = instance_.as<GTNode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const GE &op) {
    CHECK(instance_->nodeType() == ASTNodeType::GE);
    auto instance = instance_.as<GENode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const EQ &op) {
    CHECK(instance_->nodeType() == ASTNodeType::EQ);
    auto instance = instance_.as<EQNode>();
    TRY_RECURSE(op->lhs_, instance->lhs_);
    TRY_RECURSE(op->rhs_, instance->rhs_);
    if (!isMatched_) {
        isMatched_ = true;
        RECURSE(op->lhs_, instance->rhs_);
        RECURSE(op->rhs_, instance->lhs_);
    }
}

void MatchVisitor::visit(const NE &op) {
    CHECK(instance_->nodeType() == ASTNodeType::NE);
    auto instance = instance_.as<NENode>();
    TRY_RECURSE(op->lhs_, instance->lhs_);
    TRY_RECURSE(op->rhs_, instance->rhs_);
    if (!isMatched_) {
        isMatched_ = true;
        RECURSE(op->lhs_, instance->rhs_);
        RECURSE(op->rhs_, instance->lhs_);
    }
}

void MatchVisitor::visit(const LAnd &op) {
    CHECK(instance_->nodeType() == ASTNodeType::LAnd);
    auto instance = instance_.as<LAndNode>();
    TRY_RECURSE(op->lhs_, instance->lhs_);
    TRY_RECURSE(op->rhs_, instance->rhs_);
    if (!isMatched_) {
        isMatched_ = true;
        RECURSE(op->lhs_, instance->rhs_);
        RECURSE(op->rhs_, instance->lhs_);
    }
}

void MatchVisitor::visit(const LOr &op) {
    CHECK(instance_->nodeType() == ASTNodeType::LOr);
    auto instance = instance_.as<LOrNode>();
    TRY_RECURSE(op->lhs_, instance->lhs_);
    TRY_RECURSE(op->rhs_, instance->rhs_);
    if (!isMatched_) {
        isMatched_ = true;
        RECURSE(op->lhs_, instance->rhs_);
        RECURSE(op->rhs_, instance->lhs_);
    }
}

void MatchVisitor::visit(const LNot &op) {
    CHECK(instance_->nodeType() == ASTNodeType::LNot);
    auto instance = instance_.as<LNotNode>();
    RECURSE(op->expr_, instance->expr_);
}

void MatchVisitor::visit(const Sqrt &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Sqrt);
    auto instance = instance_.as<SqrtNode>();
    RECURSE(op->expr_, instance->expr_);
}

void MatchVisitor::visit(const Exp &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Exp);
    auto instance = instance_.as<ExpNode>();
    RECURSE(op->expr_, instance->expr_);
}

void MatchVisitor::visit(const Square &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Square);
    auto instance = instance_.as<SquareNode>();
    RECURSE(op->expr_, instance->expr_);
}

void MatchVisitor::visit(const Sigmoid &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Sigmoid);
    auto instance = instance_.as<SigmoidNode>();
    RECURSE(op->expr_, instance->expr_);
}

void MatchVisitor::visit(const Tanh &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Tanh);
    auto instance = instance_.as<TanhNode>();
    RECURSE(op->expr_, instance->expr_);
}

void MatchVisitor::visit(const Abs &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Abs);
    auto instance = instance_.as<AbsNode>();
    RECURSE(op->expr_, instance->expr_);
}

void MatchVisitor::visit(const Floor &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Floor);
    auto instance = instance_.as<FloorNode>();
    RECURSE(op->expr_, instance->expr_);
}

void MatchVisitor::visit(const Ceil &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Ceil);
    auto instance = instance_.as<CeilNode>();
    RECURSE(op->expr_, instance->expr_);
}

void MatchVisitor::visit(const IfExpr &op) {
    CHECK(instance_->nodeType() == ASTNodeType::IfExpr);
    auto instance = instance_.as<IfExprNode>();
    RECURSE(op->cond_, instance->cond_);
    RECURSE(op->thenCase_, instance->thenCase_);
    RECURSE(op->elseCase_, instance->elseCase_);
}

void MatchVisitor::visit(const Cast &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Cast);
    auto instance = instance_.as<CastNode>();
    RECURSE(op->expr_, instance->expr_);
    CHECK(op->dtype_ == instance->dtype_);
}

void MatchVisitor::visit(const For &op) {
    CHECK(instance_->nodeType() == ASTNodeType::For);
    auto instance = instance_.as<ForNode>();
    CHECK(matchName(op->iter_, instance->iter_));
    RECURSE(op->begin_, instance->begin_);
    RECURSE(op->end_, instance->end_);
    RECURSE(op->step_, instance->step_);
    RECURSE(op->body_, instance->body_);
    clearName(op->iter_);
}

void MatchVisitor::visit(const If &op) {
    CHECK(instance_->nodeType() == ASTNodeType::If);
    auto instance = instance_.as<IfNode>();
    RECURSE(op->cond_, instance->cond_);
    RECURSE(op->thenCase_, instance->thenCase_);
    CHECK(op->elseCase_.isValid() == instance->elseCase_.isValid());
    if (op->elseCase_.isValid()) {
        RECURSE(op->elseCase_, instance->elseCase_);
    }
}

void MatchVisitor::visit(const Assert &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Assert);
    auto instance = instance_.as<AssertNode>();
    RECURSE(op->cond_, instance->cond_);
    RECURSE(op->body_, instance->body_);
}

void MatchVisitor::visit(const Intrinsic &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Intrinsic);
    auto instance = instance_.as<IntrinsicNode>();
    CHECK(op->format_ == instance->format_);
    CHECK(op->params_.size() == instance->params_.size());
    for (auto &&[oParam, iParam] : iter::zip(op->params_, instance->params_)) {
        RECURSE(oParam, iParam);
    }
    CHECK(op->retType_ == instance->retType_);
}

void MatchVisitor::visit(const Eval &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Eval);
    auto instance = instance_.as<EvalNode>();
    RECURSE(op->expr_, instance->expr_);
}

void MatchVisitor::visit(const MatMul &op) {
    CHECK(instance_->nodeType() == ASTNodeType::MatMul);
    auto instance = instance_.as<MatMulNode>();
    RECURSE(op->a_, instance->a_);
    RECURSE(op->b_, instance->b_);
    RECURSE(op->c_, instance->c_);
    RECURSE(op->alpha_, instance->alpha_);
    RECURSE(op->beta_, instance->beta_);
    RECURSE(op->m_, instance->m_);
    RECURSE(op->k_, instance->k_);
    RECURSE(op->n_, instance->n_);
    RECURSE(op->lda_, instance->lda_);
    RECURSE(op->ldb_, instance->ldb_);
    RECURSE(op->ldc_, instance->ldc_);
    RECURSE(op->stridea_, instance->stridea_);
    RECURSE(op->strideb_, instance->strideb_);
    RECURSE(op->stridec_, instance->stridec_);
    RECURSE(op->batchSize_, instance->batchSize_);
    CHECK(op->aIsRowMajor_ == instance->aIsRowMajor_);
    CHECK(op->bIsRowMajor_ == instance->bIsRowMajor_);
    CHECK(op->cIsRowMajor_ == instance->cIsRowMajor_);
    RECURSE(op->equivalent_, instance->equivalent_);
}

bool match(const Stmt &_pattern, const Stmt &_instance) {
    auto pattern = flattenStmtSeq(_pattern);
    auto instance = flattenStmtSeq(_instance);
    MatchVisitor visitor(instance);
    visitor(pattern);
    return visitor.isMatched();
}

} // namespace ir
