#ifndef FREE_TENSOR_SHRINK_VAR_H
#define FREE_TENSOR_SHRINK_VAR_H

#include <unordered_map>

#include <analyze/comp_access_bound.h>
#include <container_utils.h>
#include <func.h>
#include <mutator.h>

namespace freetensor {

class ShrinkVar : public Mutator {
    std::unordered_map<std::string, std::vector<Expr>> lower_, upper_;
    const std::unordered_map<ID, AccessBound> &newRange_;

  public:
    ShrinkVar(const std::unordered_map<ID, AccessBound> &newRange)
        : newRange_(newRange) {}

  private:
    template <class T> T modifyAccess(const T &op) {
        if (lower_.count(op->var_)) {
            auto &&offset = lower_.at(op->var_);
            ASSERT(offset.size() == op->indices_.size());
            for (auto &&[idx, off] : views::zip(op->indices_, offset)) {
                if (off.isValid()) {
                    idx = makeSub(idx, off);
                }
            }
        }
        return op;
    }

    template <class T> Stmt addCheck(const T &oldOp, const T &op) {
        // We add check w.r.t oldOp because it is simplier, which brings less
        // redundancy to pass/simplify
        Expr guard;
        if (upper_.count(op->var_)) {
            auto &&upper = upper_.at(op->var_);
            ASSERT(upper.size() == op->indices_.size());
            for (auto &&[idx, u] : views::zip(oldOp->indices_, upper)) {
                if (u.isValid()) {
                    guard = guard.isValid() ? makeLAnd(guard, makeLE(idx, u))
                                            : makeLE(idx, u);
                }
            }
        }
        if (lower_.count(op->var_)) {
            auto &&lower = lower_.at(op->var_);
            ASSERT(lower.size() == op->indices_.size());
            for (auto &&[idx, l] : views::zip(oldOp->indices_, lower)) {
                if (l.isValid()) {
                    guard = guard.isValid() ? makeLAnd(guard, makeGE(idx, l))
                                            : makeGE(idx, l);
                }
            }
        }
        return guard.isValid() ? makeIf(std::move(guard), op) : op;
    }

  protected:
    Stmt visit(const VarDef &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
};

/**
 * Make the shape of a variable smaller if some elements are not used
 *
 * If you don't want to shrink some variables, please set VarDefNode::pinned_.
 * I/O variables will not be shrinked
 *
 * Set the `varDefId` parameter to shrink only one variable
 *
 * @{
 */
Stmt shrinkVar(const Stmt &op);
Stmt shrinkSingleVar(const Stmt &op, const ID &varDefId);
/** @} */

DEFINE_PASS_FOR_FUNC(shrinkVar)

} // namespace freetensor

#endif // FREE_TENSOR_SHRINK_VAR_H
