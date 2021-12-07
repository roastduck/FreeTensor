#ifndef MAKE_REDUCTION_H
#define MAKE_REDUCTION_H

#include <unordered_set>

#include <analyze/all_reads.h>
#include <func.h>
#include <mutator.h>

namespace ir {

class MakeReduction : public Mutator {
    const std::unordered_set<ReduceOp> &types_;
    bool canonicalOnly_;

  public:
    MakeReduction(const std::unordered_set<ReduceOp> &types, bool canonicalOnly)
        : types_(types), canonicalOnly_(canonicalOnly) {}

  private:
    bool isSameElem(const Store &s, const Load &l);

    template <class Node> Stmt doMake(Store op, ReduceOp reduceOp) {
        if (types_.count(reduceOp)) {
            auto expr = op->expr_.as<Node>();
            if (expr->lhs_->nodeType() == ASTNodeType::Load &&
                isSameElem(op, expr->lhs_.template as<LoadNode>()) &&
                (!canonicalOnly_ || !allReads(expr->rhs_).count(op->var_))) {
                return makeReduceTo(op->id(), op->var_, op->indices_, reduceOp,
                                    expr->rhs_, false);
            }
            if (expr->rhs_->nodeType() == ASTNodeType::Load &&
                isSameElem(op, expr->rhs_.template as<LoadNode>()) &&
                (!canonicalOnly_ || !allReads(expr->lhs_).count(op->var_))) {
                return makeReduceTo(op->id(), op->var_, op->indices_, reduceOp,
                                    expr->lhs_, false);
            }
        }
        return op;
    }

  protected:
    Stmt visit(const Store &op) override;
};

/**
 * Transform things like a = a + b into a += b
 *
 * This is to make the dependency analysis more accurate
 *
 * @param types : Only transform these types of reductions
 * @param canonicalOnly : True to avoid cyclic reductions like a += a + b
 */
inline Stmt makeReduction(const Stmt &op,
                          const std::unordered_set<ReduceOp> &types,
                          bool canonicalOnly = false) {
    return MakeReduction(types, canonicalOnly)(op);
}

inline Stmt makeReduction(const Stmt &op) {
    return makeReduction(
        op, {ReduceOp::Add, ReduceOp::Mul, ReduceOp::Min, ReduceOp::Max});
}

DEFINE_PASS_FOR_FUNC(makeReduction)

} // namespace ir

#endif // MAKE_REDUCTION_H
