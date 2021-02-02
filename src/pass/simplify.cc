#include <functional>
#include <sstream>

#include <analyze/hash.h>
#include <analyze/normalize.h>
#include <except.h>
#include <pass/disambiguous.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/simplify.h>

namespace ir {

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

void FindInnerMostScope::visit(const Var &op) {
    Visitor::visit(op);
    innerMost_ = std::max(innerMost_, varScope_.at(op->name_));
}

void FindInnerMostScope::visit(const Load &op) {
    Visitor::visit(op);
    innerMost_ = std::max(innerMost_, varScope_.at(op->var_));
}

int findInnerMostScope(const std::unordered_map<std::string, int> &varScope,
                       const Expr &op) {
    FindInnerMostScope visitor(varScope);
    visitor(op);
    return visitor.innnerMost();
}

uint64_t SimplifyPass::getHash(const Expr &op) {
    if (hash_.count(op)) { // maybe not, beacuse Mutator::visit
        return hash_.at(op);
    } else {
        return ::ir::getHash(op);
    }
}

Expr SimplifyPass::visit(const Div &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Div);
    auto op = __op.as<DivNode>();
    if (op->lhs_->nodeType() == ASTNodeType::IntConst &&
        op->rhs_->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(op->lhs_.as<IntConstNode>()->val_ /
                            op->rhs_.as<IntConstNode>()->val_);
    }
    return op;
}

Expr SimplifyPass::visit(const Mod &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mod);
    auto op = __op.as<DivNode>();
    if (op->lhs_->nodeType() == ASTNodeType::IntConst &&
        op->rhs_->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(op->lhs_.as<IntConstNode>()->val_ %
                            op->rhs_.as<IntConstNode>()->val_);
    }
    return op;
}

Expr SimplifyPass::visit(const Min &op) {
    ASSERT(op->info_norm_form_.isValid());
    if (checkUpperCmp0(op, std::less_equal<int>())) {
        isFixPoint_ = false;
        return (*this)(op->lhs_);
    }
    if (checkLowerCmp0(op, std::greater_equal<int>())) {
        isFixPoint_ = false;
        return (*this)(op->rhs_);
    }
    return Mutator::visit(op);
}

Expr SimplifyPass::visit(const Max &op) {
    ASSERT(op->info_norm_form_.isValid());
    if (checkLowerCmp0(op, std::greater_equal<int>())) {
        isFixPoint_ = false;
        return (*this)(op->lhs_);
    }
    if (checkUpperCmp0(op, std::less_equal<int>())) {
        isFixPoint_ = false;
        return (*this)(op->rhs_);
    }
    return Mutator::visit(op);
}

Expr SimplifyPass::visit(const LT &op) {
    ASSERT(op->info_norm_form_.isValid());
    if (checkUpperCmp0(op, std::less<int>())) {
        isFixPoint_ = false;
        return makeIntConst(1);
    }
    if (checkLowerCmp0(op, std::greater_equal<int>())) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    return Mutator::visit(op);
}

Expr SimplifyPass::visit(const LE &op) {
    ASSERT(op->info_norm_form_.isValid());
    if (checkUpperCmp0(op, std::less_equal<int>())) {
        isFixPoint_ = false;
        return makeIntConst(1);
    }
    if (checkLowerCmp0(op, std::greater<int>())) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    return Mutator::visit(op);
}

Expr SimplifyPass::visit(const GT &op) {
    ASSERT(op->info_norm_form_.isValid());
    if (checkUpperCmp0(op, std::less_equal<int>())) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    if (checkLowerCmp0(op, std::greater<int>())) {
        isFixPoint_ = false;
        return makeIntConst(1);
    }
    return Mutator::visit(op);
}

Expr SimplifyPass::visit(const GE &op) {
    ASSERT(op->info_norm_form_.isValid());
    if (checkUpperCmp0(op, std::less<int>())) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    if (checkLowerCmp0(op, std::greater_equal<int>())) {
        isFixPoint_ = false;
        return makeIntConst(1);
    }
    return Mutator::visit(op);
}

Expr SimplifyPass::visit(const EQ &op) {
    ASSERT(op->info_norm_form_.isValid());
    if (checkUpperCmp0(op, std::less<int>())) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    if (checkLowerCmp0(op, std::greater<int>())) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    if (upper_.count(op->info_norm_form_) &&
        lower_.count(op->info_norm_form_)) {
        for (auto &&upper : upper_.at(op->info_norm_form_)) {
            if (upper.expr_->nodeType() == ASTNodeType::IntConst &&
                upper.expr_.as<IntConstNode>()->val_ == 0) {
                for (auto &&lower : lower_.at(op->info_norm_form_)) {
                    if (lower.expr_->nodeType() == ASTNodeType::IntConst &&
                        lower.expr_.as<IntConstNode>()->val_ == 0) {
                        isFixPoint_ = false;
                        return makeIntConst(1);
                    }
                }
            }
        }
    }
    return Mutator::visit(op);
}

Expr SimplifyPass::visit(const NE &op) {
    ASSERT(op->info_norm_form_.isValid());
    if (checkUpperCmp0(op, std::less<int>())) {
        isFixPoint_ = false;
        return makeIntConst(1);
    }
    if (checkLowerCmp0(op, std::greater<int>())) {
        isFixPoint_ = false;
        return makeIntConst(1);
    }
    if (upper_.count(op->info_norm_form_) &&
        lower_.count(op->info_norm_form_)) {
        for (auto &&upper : upper_.at(op->info_norm_form_)) {
            if (upper.expr_->nodeType() == ASTNodeType::IntConst &&
                upper.expr_.as<IntConstNode>()->val_ == 0) {
                for (auto &&lower : lower_.at(op->info_norm_form_)) {
                    if (lower.expr_->nodeType() == ASTNodeType::IntConst &&
                        lower.expr_.as<IntConstNode>()->val_ == 0) {
                        isFixPoint_ = false;
                        return makeIntConst(0);
                    }
                }
            }
        }
    }
    return Mutator::visit(op);
}

Expr SimplifyPass::visit(const LAnd &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LAnd);
    auto op = __op.as<LAndNode>();
    if (op->lhs_->nodeType() == ASTNodeType::IntConst) {
        return op->lhs_.as<IntConstNode>()->val_ ? op->rhs_ : makeIntConst(0);
    }
    if (op->rhs_->nodeType() == ASTNodeType::IntConst) {
        return op->rhs_.as<IntConstNode>()->val_ ? op->lhs_ : makeIntConst(0);
    }
    return op;
}

Expr SimplifyPass::visit(const LOr &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LOr);
    auto op = __op.as<LOrNode>();
    if (op->lhs_->nodeType() == ASTNodeType::IntConst) {
        return op->lhs_.as<IntConstNode>()->val_ ? makeIntConst(1) : op->rhs_;
    }
    if (op->rhs_->nodeType() == ASTNodeType::IntConst) {
        return op->rhs_.as<IntConstNode>()->val_ ? makeIntConst(1) : op->lhs_;
    }
    return op;
}

Expr SimplifyPass::visit(const LNot &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LNot);
    auto op = __op.as<LNotNode>();
    if (op->expr_->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(!op->expr_.as<IntConstNode>()->val_);
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
    auto __op = Mutator::visit(_op);
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
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    varScope_.erase(_op->iter_), curScope_--;

    if (isEmptyStmt(op->body_)) {
        return makeStmtSeq("", {});
    }
    if (op->info_len_->nodeType() == ASTNodeType::IntConst) {
        auto len = op->info_len_.as<IntConstNode>()->val_;
        if (len == 1) {
            return op->body_;
        }
        if (len <= 0) {
            return makeStmtSeq("", {});
        }
    }

    return op;
}

Stmt SimplifyPass::visit(const If &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.as<IfNode>();
    if (isEmptyStmt(op->thenCase_) && isEmptyStmt(op->elseCase_)) {
        return makeStmtSeq("", {});
    }
    if (op->cond_->nodeType() == ASTNodeType::IntConst) {
        isFixPoint_ = false;
        if (op->cond_.as<IntConstNode>()->val_) {
            return op->thenCase_;
        } else {
            if (op->elseCase_.isValid()) {
                return op->elseCase_;
            } else {
                return makeStmtSeq("", {});
            }
        }
    }
    return op;
}

Stmt SimplifyPass::visit(const Assert &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Assert);
    auto op = __op.as<AssertNode>();
    if (op->cond_->nodeType() == ASTNodeType::IntConst) {
        isFixPoint_ = false;
        if (op->cond_.as<IntConstNode>()->val_) {
            return op->body_;
        } else {
            std::ostringstream os;
            // Print the unchanged _op
            os << "Assertion always false: " << _op;
            throw InvalidSchedule(os.str());
        }
    }
    return op;
}

Stmt simplifyPass(const Stmt &op) {
    return flattenStmtSeq(std::get<0>(simplifyAndGetBounds(op)));
}

std::tuple<Stmt, SimplifyPass::BoundsMap, SimplifyPass::BoundsMap>
simplifyAndGetBounds(const Stmt &_op) {
    Stmt op = normalize(_op);
    for (int i = 0;; i++) {
        op = disambiguous(op);

        auto hash = getHashMap(op);

        AnalyzeBounds analyzeBounds(hash);
        analyzeBounds(op);
        auto &&lower = analyzeBounds.lower();
        auto &&upper = analyzeBounds.upper();

        SimplifyPass simplifyVisitor(hash, lower, upper);
        auto newOp = simplifyVisitor(op);
        if (simplifyVisitor.isFixPoint() || i > 100) {
            if (i > 100) {
                WARNING("SimplifyPass iterates over 100 rounds. Maybe there is "
                        "a bug");
            }
            return {op, lower, upper}; // return the old op, or the lower /
                                       // upper will be invalid
        } else {
            op = newOp;
        }
    }
}

} // namespace ir

