#include <debug/match_ast.h>

namespace ir {

bool MatchVisitor::matchName(const std::string &thisName,
                             const std::string &otherName) {
    if (!nameMap_.count(thisName)) {
        nameMap_[thisName] = otherName;
        return true;
    }
    return nameMap_.at(thisName) == otherName;
}

#define CHECK(expr)                                                            \
    if (!(expr)) {                                                             \
        isMatched_ = false;                                                    \
        return;                                                                \
    }

#define RECURSE(lexpr, rexpr)                                                  \
    {                                                                          \
        auto oldInstance = instance_;                                          \
        instance_ = rexpr;                                                     \
        (*this)(lexpr);                                                        \
        instance_ = oldInstance;                                               \
        if (!isMatched_) {                                                     \
            return;                                                            \
        }                                                                      \
    }

void MatchVisitor::visit(const StmtSeq &op) {
    CHECK(instance_->nodeType() == ASTNodeType::StmtSeq);
    auto instance = instance_.as<StmtSeqNode>();
    CHECK(op->stmts_.size() == instance->stmts_.size());
    for (size_t i = 0, iEnd = op->stmts_.size(); i < iEnd; i++) {
        RECURSE(op->stmts_[i], instance->stmts_[i]);
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
    for (size_t i = 0, iEnd = lshape.size(); i < iEnd; i++) {
        RECURSE(lshape[i], rshape[i]);
    }
    RECURSE(op->body_, instance->body_);
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
    for (size_t i = 0, iEnd = op->indices_.size(); i < iEnd; i++) {
        RECURSE(op->indices_[i], instance->indices_[i]);
    }
    RECURSE(op->expr_, instance->expr_);
}

void MatchVisitor::visit(const Load &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Load);
    auto instance = instance_.as<LoadNode>();
    CHECK(matchName(op->var_, instance->var_));
    for (size_t i = 0, iEnd = op->indices_.size(); i < iEnd; i++) {
        RECURSE(op->indices_[i], instance->indices_[i]);
    }
}

void MatchVisitor::visit(const ReduceTo &op) {
    CHECK(instance_->nodeType() == ASTNodeType::ReduceTo);
    auto instance = instance_.as<ReduceToNode>();
    CHECK(matchName(op->var_, instance->var_));
    for (size_t i = 0, iEnd = op->indices_.size(); i < iEnd; i++) {
        RECURSE(op->indices_[i], instance->indices_[i]);
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

void MatchVisitor::visit(const Add &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Add);
    auto instance = instance_.as<AddNode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
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
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const Div &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Div);
    auto instance = instance_.as<DivNode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const Mod &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Mod);
    auto instance = instance_.as<ModNode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const Min &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Min);
    auto instance = instance_.as<MinNode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const Max &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Max);
    auto instance = instance_.as<MaxNode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
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
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const NE &op) {
    CHECK(instance_->nodeType() == ASTNodeType::NE);
    auto instance = instance_.as<NENode>();
    RECURSE(op->lhs_, instance->lhs_);
    RECURSE(op->rhs_, instance->rhs_);
}

void MatchVisitor::visit(const Not &op) {
    CHECK(instance_->nodeType() == ASTNodeType::Not);
    auto instance = instance_.as<NotNode>();
    RECURSE(op->expr_, instance->expr_);
}

void MatchVisitor::visit(const For &op) {
    CHECK(instance_->nodeType() == ASTNodeType::For);
    auto instance = instance_.as<ForNode>();
    CHECK(matchName(op->iter_, instance->iter_));
    RECURSE(op->begin_, instance->begin_);
    RECURSE(op->end_, instance->end_);
    RECURSE(op->body_, instance->body_);
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

bool match(const AST &pattern, const AST &instance) {
    MatchVisitor visitor(instance);
    visitor(pattern);
    return visitor.isMatched();
}

} // namespace ir

