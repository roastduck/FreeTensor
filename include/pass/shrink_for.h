#ifndef SHRINK_FOR_H
#define SHRINK_FOR_H

#include <pass/simplify.h>

namespace ir {

class ShrinkFor : public SimplifyPass {
    std::unordered_map<uint64_t, std::pair<Expr, Expr>> newRange_;
    bool keepConst_;

  public:
    ShrinkFor(bool keepConst) : keepConst_(keepConst) {}

  private:
    Expr simplifyExpr(const Expr &expr);

    template <class T> Stmt visitSideEffect(const T &op) {
        auto ret = SimplifyPass::visit(op);
        for (auto &&item : transients()) {
            if (newRange_.count(item.first)) {
                newRange_[item.first] = std::make_pair(
                    makeMin(newRange_[item.first].first, item.second.first),
                    makeMax(newRange_[item.first].second, item.second.second));
            } else {
                newRange_[item.first] = item.second;
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
