#include <schedule/unroll.h>

namespace ir {

Stmt BackUnroll::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (op->id() == loop_) {
        if (op->len_->nodeType() == ASTNodeType::IntConst) {
            op->unroll_ = true;
            done_ = true;
        } else {
            throw InvalidSchedule("Length of the loop should be constant.");
        }
    }
    return op;
}

Stmt ImmediateUnroll::visitStmt(
    const Stmt &op, const std::function<Stmt(const Stmt &)> &visitNode) {
    auto ret = Mutator::visitStmt(op, visitNode);
    if (!iter_.empty()) {
        ret->setId(ret->id() + "." + std::to_string(curIter_));
    }
    return ret;
}

Expr ImmediateUnroll::visit(const Var &op) {
    if (op->name_ == iter_) {
        return makeIntConst(curIter_);
    } else {
        return Mutator::visit(op);
    }
}

Stmt ImmediateUnroll::visit(const For &op) {
    if (op->id() == loop_) {
        if (op->len_->nodeType() == ASTNodeType::IntConst) {
            auto len = op->len_.as<IntConstNode>()->val_;
            std::vector<Stmt> stmts;
            iter_ = op->iter_;
            begin_ = op->begin_;
            for (curIter_ = 0; curIter_ < len; curIter_++) {
                stmts.emplace_back((*this)(op->body_));
            }
            begin_ = nullptr;
            iter_.clear();
            done_ = true;
            return makeStmtSeq("", std::move(stmts));
        } else {
            throw InvalidSchedule("Length of the loop should be constant.");
        }
    } else {
        return Mutator::visit(op);
    }
}

} // namespace ir
