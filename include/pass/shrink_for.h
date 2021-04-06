#ifndef SHRINK_FOR_H
#define SHRINK_FOR_H

#include <pass/simplify.h>

namespace ir {

class ShrinkFor : public SimplifyPass {
    std::unordered_map<uint64_t, std::pair<Expr, Expr>> newRange_;
    std::vector<Var> iterStack_;
    bool keepConst_;

  public:
    ShrinkFor(bool keepConst) : keepConst_(keepConst) {}

  private:
    Expr simplifyExpr(const Expr &expr);

    template <class T> Stmt visitSideEffect(const T &op) {
        auto ret = SimplifyPass::visit(op);
        for (auto var : iterStack_) {
            auto hash = getHash(var);
            if (auto bound = transient(var); bound.isValid()) {
                if (newRange_.count(hash)) {
                    newRange_[hash] = std::make_pair(
                        makeMin(newRange_[hash].first, bound->first),
                        makeMax(newRange_[hash].second, bound->second));
                } else {
                    newRange_[hash] = *bound;
                }
            }
        }
        return ret;
    }

  protected:
    using SimplifyPass::visit;

    Stmt visit(const Store &op) override { return visitSideEffect(op); }
    Stmt visit(const ReduceTo &op) override { return visitSideEffect(op); }
    // TODO: Also for Eval with side effect
    Stmt visit(const For &op) override;
};

/**
 * Increase the begin and decrease the end index, to remove redundant iterations
 * from For loops
 *
 * @param keepConst : If true, do not transform loops to have variable begins
 * and ends.
 */
Stmt shrinkFor(const Stmt &op, bool keepConst = false);

} // namespace ir

#endif // SHRINK_FOR_H
