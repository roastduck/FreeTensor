#include <pass/remove_writes.h>
#include <pass/simplify.h>
#include <schedule/unroll.h>

namespace freetensor {

Stmt BackUnroll::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (op->id() == loop_) {
        if (op->len_->nodeType() == ASTNodeType::IntConst) {
            op->property_->unroll_ = true;
            done_ = true;
        } else {
            throw InvalidSchedule("Length of the loop should be constant.");
        }
    }
    return op;
}

Stmt ImmediateUnroll::visitStmt(const Stmt &op) {
    auto ret = Mutator::visitStmt(op);
    if (!iter_.empty()) {
        ret->setId();
        ret->metadata() =
            makeMetadata("unroll." + std::to_string(curIter_), ret);
    }
    return ret;
}

Expr ImmediateUnroll::visit(const Var &op) {
    if (op->name_ == iter_) {
        return makeAdd(begin_, makeMul(makeIntConst(curIter_), step_));
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
            begin_ = op->begin_, step_ = op->step_;
            for (curIter_ = 0; curIter_ < len; curIter_++) {
                stmts.emplace_back((*this)(op->body_));
            }
            begin_ = step_ = nullptr;
            iter_.clear();
            done_ = true;
            return makeStmtSeq(std::move(stmts));
        } else {
            throw InvalidSchedule("Length of the loop should be constant.");
        }
    } else {
        return Mutator::visit(op);
    }
}

Stmt unroll(const Stmt &_ast, const ID &loop, bool immediate) {
    auto ast =
        simplify(_ast); // Make things like range(n, n + 4) constant ranges
    bool done = false;
    if (immediate) {
        ImmediateUnroll mutator(loop);
        ast = mutator(ast);
        done = mutator.done();
        ast = removeWrites(ast);
    } else {
        BackUnroll mutator(loop);
        ast = mutator(ast);
        done = mutator.done();
    }
    if (!done) {
        throw InvalidSchedule("Loop " + toString(loop) + " not found");
    }
    return ast;
}

} // namespace freetensor
