#ifndef SHRINK_VAR_H
#define SHRINK_VAR_H

#include <unordered_map>

#include <analyze/comp_access_bound.h>
#include <func.h>
#include <mutator.h>

namespace ir {

class ShrinkVar : public Mutator {
    std::unordered_map<std::string, std::vector<Expr>> offset_;
    const std::unordered_map<std::string, AccessBound> &newRange_;

  public:
    ShrinkVar(const std::unordered_map<std::string, AccessBound> &newRange)
        : newRange_(newRange) {}

  private:
    template <class T> T modifyAccess(const T &op) {
        if (offset_.count(op->var_)) {
            auto &&offset = offset_.at(op->var_);
            ASSERT(offset.size() == op->indices_.size());
            for (size_t i = 0, iEnd = offset.size(); i < iEnd; i++) {
                op->indices_[i] = makeSub(op->indices_[i], offset[i]);
            }
        }
        return op;
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
 * If you don't want to shrink some variables, please set VarDefNode::pinned_
 */
Stmt shrinkVar(const Stmt &op);

/**
 * A variant of shrinkVar that shrinks only one variable only
 */
Stmt shrinkSingleVar(const Stmt &op, const std::string &varDefId);

DEFINE_PASS_FOR_FUNC(shrinkVar)

} // namespace ir

#endif // SHRINK_VAR_H
