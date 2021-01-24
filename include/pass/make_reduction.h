#ifndef MAKE_REDUCTION_H
#define MAKE_REDUCTION_H

#include <mutator.h>

namespace ir {

/**
 * Transform things like a = a + b into a += b
 *
 * This is to make the dependency analysis more accurate
 */
class MakeReduction : public Mutator {
  private:
    bool isSameElem(const Store &s, const Load &l);

    template <class Node> Stmt doMake(Store op, ReduceOp reduceOp) {
        auto expr = op->expr_.as<Node>();
        if (expr->lhs_->nodeType() == ASTNodeType::Load &&
            isSameElem(op, expr->lhs_.template as<LoadNode>())) {
            return makeReduceTo(op->id(), op->var_, op->indices_, reduceOp,
                                expr->rhs_);
        }
        if (expr->rhs_->nodeType() == ASTNodeType::Load &&
            isSameElem(op, expr->rhs_.template as<LoadNode>())) {
            return makeReduceTo(op->id(), op->var_, op->indices_, reduceOp,
                                expr->lhs_);
        }
        return op;
    }

  protected:
    Stmt visit(const Store &op) override;
};

inline Stmt makeReduction(const Stmt &op) { return MakeReduction()(op); }

} // namespace ir

#endif // MAKE_REDUCTION_H
