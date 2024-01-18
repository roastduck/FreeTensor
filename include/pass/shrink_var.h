#ifndef FREE_TENSOR_SHRINK_VAR_H
#define FREE_TENSOR_SHRINK_VAR_H

#include <unordered_map>

#include <analyze/comp_access_bound.h>
#include <container_utils.h>
#include <func.h>
#include <mutator.h>

namespace freetensor {

/**
 * Main mutator for shrinking variables
 *
 * This mutator modifies the shape of each variable to be the upper bound
 * expression minus the lower bound expression plus one, with respect to each
 * access of the variable.
 */
class ShrinkVar : public Mutator {
    // Bound considering the old shape. Used for preventing make the shape even
    // larger after shrinking
    const std::unordered_map<ID, AccessBound> &newRangeWithShape_;

    // Bound without considering the old shape. Used for preventing redundant
    // guards for maybe-unsafe user code
    const std::unordered_map<ID, AccessBound> &newRangeWithoutShape_;

    bool guardReads_;

    std::unordered_map<std::string, std::vector<Expr>> lowerWithShape_,
        upperWithShape_;
    std::unordered_map<std::string, std::vector<Expr>> lowerWithoutShape_,
        upperWithoutShape_;
    std::unordered_map<ID, Expr> guards_;

  public:
    ShrinkVar(const std::unordered_map<ID, AccessBound> &newRangeWithShape,
              const std::unordered_map<ID, AccessBound> &newRangeWithoutShape,
              bool guardReads = false)
        : newRangeWithShape_(newRangeWithShape),
          newRangeWithoutShape_(newRangeWithoutShape), guardReads_(guardReads) {
    }

  private:
    template <class T> T modifyAccess(const T &op) {
        if (lowerWithoutShape_.count(op->var_)) {
            auto &&offset = lowerWithoutShape_.at(op->var_);
            ASSERT(offset.size() == op->indices_.size());
            for (auto &&[idx, off] : views::zip(op->indices_, offset)) {
                if (off.isValid()) {
                    idx = makeSub(idx, off);
                }
            }
        }
        return op;
    }

    template <class T> void addGuard(const T &oldOp, const T &op) {
        // We add check w.r.t oldOp because it is simplier, which brings less
        // redundancy to pass/simplify
        Expr guard;
        if (upperWithoutShape_.count(op->var_)) {
            auto &&upper = upperWithoutShape_.at(op->var_);
            ASSERT(upper.size() == op->indices_.size());
            for (auto &&[idx, u] : views::zip(oldOp->indices_, upper)) {
                if (u.isValid()) {
                    guard = guard.isValid() ? makeLAnd(guard, makeLE(idx, u))
                                            : makeLE(idx, u);
                }
            }
        }
        if (lowerWithoutShape_.count(op->var_)) {
            auto &&lower = lowerWithoutShape_.at(op->var_);
            ASSERT(lower.size() == op->indices_.size());
            for (auto &&[idx, l] : views::zip(oldOp->indices_, lower)) {
                if (l.isValid()) {
                    guard = guard.isValid() ? makeLAnd(guard, makeGE(idx, l))
                                            : makeGE(idx, l);
                }
            }
        }
        if (guard.isValid()) {
            Stmt s;
            if constexpr (std::is_base_of_v<StmtNode, typename T::Object>) {
                s = oldOp;
            } else if constexpr (std::is_base_of_v<ExprNode,
                                                   typename T::Object>) {
                s = oldOp->parentStmt();
            } else {
                ASSERT(false);
            }
            guards_[s->id()] = guards_[s->id()].isValid()
                                   ? makeLAnd(guards_[s->id()], guard)
                                   : guard;
        }
    }

  protected:
    Stmt visitStmt(const Stmt &s) override;
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
