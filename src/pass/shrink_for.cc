#include <math/min_max.h>
#include <pass/shrink_for.h>
#include <pass/simplify.h>
#include <pass/z3_simplify.h>

namespace freetensor {

void CheckSideEffect::visit(const Store &op) { hasSideEffect_ = true; }

void CheckSideEffect::visit(const ReduceTo &op) { hasSideEffect_ = true; }

void CheckSideEffect::visit(const Intrinsic &op) {
    if (op->hasSideEffect_) {
        hasSideEffect_ = true;
    } else {
        Visitor::visit(op);
    }
}

Stmt ShrinkFor::visitStmt(const Stmt &stmt) {
    auto ret = BaseClass::visitStmt(stmt);

    CheckSideEffect checker;
    switch (stmt->nodeType()) {
    case ASTNodeType::Store:
    case ASTNodeType::ReduceTo:
    case ASTNodeType::Eval:
        checker(stmt);
        break;
    case ASTNodeType::If:
        checker(stmt.as<IfNode>()->cond_);
        break;
    case ASTNodeType::Assert:
        checker(stmt.as<AssertNode>()->cond_);
        break;
    case ASTNodeType::For:
        checker(stmt.as<ForNode>()->begin_);
        checker(stmt.as<ForNode>()->end_);
        checker(stmt.as<ForNode>()->step_);
        break;
    case ASTNodeType::VarDef:
        for (auto &&dim : stmt.as<VarDefNode>()->buffer_->tensor()->shape()) {
            checker(dim);
        }
        break;
    default:;
    }
    if (checker.hasSideEffect()) {
        for (auto &&[_var, _names] : views::zip(iterStack_, namesStack_)) {
            auto &&names = _names;

            // Trigger recomputing in analyze/comp_unique_bounds
            auto var = deepCopy(_var).as<VarNode>();

            std::vector<Expr> lower, upper;
            for (auto &&b : bound_.getDefinedLower(var, names)) {
                lower.emplace_back(b.expr());
            }
            for (auto &&b : bound_.getDefinedUpper(var, names)) {
                upper.emplace_back(b.expr());
            }
            newRange_[var].first.emplace_back(std::move(lower));
            newRange_[var].second.emplace_back(std::move(upper));
        }
    }

    return ret;
}

Stmt ShrinkFor::visit(const For &_op) {
    auto var = makeVar(_op->iter_).as<VarNode>();
    newRange_.erase(var);

    iterStack_.emplace_back(var);
    namesStack_.emplace_back(names());
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    namesStack_.pop_back();
    iterStack_.pop_back();

    if (!newRange_.count(var)) {
        return makeStmtSeq({});
    }
    auto lower = makeMinMax(newRange_.at(var).first);
    auto upper = makeMaxMin(newRange_.at(var).second);

    if (op->property_->unroll_) {
        // Backends do not support these loops to be of variable lengths
        lower = makeIntConst(bound_.getIntLower(lower));
        upper = makeIntConst(bound_.getIntUpper(upper));
    }

    if (op->step_->nodeType() == ASTNodeType::IntConst) {
        auto step = op->step_.as<IntConstNode>()->val_;
        if (step > 0) {
            if (lower.isValid()) {
                op->begin_ = lower;
            }
            if (upper.isValid()) {
                op->end_ = makeAdd(upper, makeIntConst(1));
            }
            op->len_ = makeCeilDiv(makeSub(op->end_, op->begin_), op->step_);
        } else if (step < 0) {
            if (upper.isValid()) {
                op->begin_ = upper;
            }
            if (lower.isValid()) {
                op->end_ = makeAdd(lower, makeIntConst(-1));
            }
            op->len_ = makeCeilDiv(makeSub(op->end_, op->begin_), op->step_);
        }
    }

    return op;
}

Stmt shrinkFor(const Stmt &_op) {
    auto op = simplify(_op); // Const prop + eliminate empty loops
    op = ShrinkFor()(op);
    return simplify(op);
}

} // namespace freetensor
