#ifndef MAKE_REDUCTION_H
#define MAKE_REDUCTION_H

#include <unordered_set>

#include <func.h>
#include <mutator.h>

namespace ir {

/**
 * Transform things like a = a + b into a += b
 *
 * This is to make the dependency analysis more accurate
 */
class MakeReduction : public Mutator {
    const std::unordered_set<ReduceOp> &types_;

  public:
    MakeReduction(const std::unordered_set<ReduceOp> &types) : types_(types) {}

  private:
    bool isSameElem(const Store &s, const Load &l);

    template <class Node> Stmt doMake(Store op, ReduceOp reduceOp) {
        if (types_.count(reduceOp)) {
            auto expr = op->expr_.as<Node>();
            if (expr->lhs_->nodeType() == ASTNodeType::Load &&
                isSameElem(op, expr->lhs_.template as<LoadNode>())) {
                return makeReduceTo(op->id(), op->var_, op->indices_, reduceOp,
                                    expr->rhs_, false);
            }
            if (expr->rhs_->nodeType() == ASTNodeType::Load &&
                isSameElem(op, expr->rhs_.template as<LoadNode>())) {
                return makeReduceTo(op->id(), op->var_, op->indices_, reduceOp,
                                    expr->lhs_, false);
            }
        }
        return op;
    }

  protected:
    Stmt visit(const Store &op) override;
};

inline Stmt makeReduction(const Stmt &op,
                          const std::unordered_set<ReduceOp> &types) {
    return MakeReduction(types)(op);
}

inline Stmt makeReduction(const Stmt &op) {
    return makeReduction(
        op, {ReduceOp::Add, ReduceOp::Mul, ReduceOp::Min, ReduceOp::Max});
}

DEFINE_PASS_FOR_FUNC(makeReduction)

} // namespace ir

#endif // MAKE_REDUCTION_H
