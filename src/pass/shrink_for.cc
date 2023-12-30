#include <math/min_max.h>
#include <pass/pb_simplify.h>
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
    if (stmt == subAST_)
        inSubAST_ = true;
    if (subAST_.isValid() && !(inSubAST_ || subASTAncestors_.contains(stmt)))
        return deepCopy(stmt);

    auto ret = BaseClass::visitStmt(stmt);

    if (stmt == subAST_)
        inSubAST_ = false;

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
            auto &&names = filterNames(_names);

            // We need linear programming from PBCompBounds, because the
            // minimum/maximum value of a linear function does not always appear
            // at the minimum/maximum points of its parameters.
            // See 2.pass/test_shrink_for.py::test_linear_bounds
            //
            // PBCompBounds requires one instance per Stmt
            CompUniqueBoundsPB bound(*this);

            // Trigger recomputing in analyze/comp_unique_bounds
            auto var = deepCopy(_var).as<VarNode>();
            newRange_[var].emplace_back(
                bound.getBound(var)->restrictScope(names));
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

    if (!filterLoop(op)) {
        return op;
    }

    if (!newRange_.count(var)) {
        return makeStmtSeq({});
    }

    // PBCompBounds requires one instance per Stmt
    CompUniqueBoundsPB bound(*this);

    auto [lower, upper] = bound.unionBounds(newRange_[var]);

    if (op->property_->unroll_) {
        // Backends do not support these loops to be of variable lengths

        lower = makeIntConst(bound.getIntLower(lower));
        upper = makeIntConst(bound.getIntUpper(upper));
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

void ShrinkFor::setSubAST(const Stmt &subAST) {
    subAST_ = subAST;
    for (Stmt s = subAST->parentStmt(); s.isValid(); s = s->parentStmt())
        subASTAncestors_.insert(s);
}

Stmt shrinkFor(const Stmt &_op, const Stmt &subAST, bool doSimplify) {
    auto op = _op;

    if (doSimplify) // Const prop + eliminate empty loops
        op = simplify(op);

    ShrinkFor shrinker;
    if (subAST.isValid())
        shrinker.setSubAST(subAST);
    op = shrinker(op);

    if (doSimplify) // Make new ranges simple + remove redundant branches
        op = simplify(z3Simplify(op));

    return op;
}

} // namespace freetensor
