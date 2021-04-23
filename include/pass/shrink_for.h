#ifndef SHRINK_FOR_H
#define SHRINK_FOR_H

#include <analyze/check_all_defined.h>
#include <pass/simplify.h>

namespace ir {

class ShrinkFor : public BuiltinSimplify {
    std::unordered_map<uint64_t, std::pair<Expr, Expr>> newRange_;
    std::vector<Var> iterStack_;
    std::vector<std::unordered_set<std::string>> defStack_;
    std::unordered_set<std::string> defs_;
    bool keepConst_;

  public:
    ShrinkFor(bool keepConst) : keepConst_(keepConst) {}

  private:
    Expr simplifyExpr(const Expr &expr);

    template <class T> Stmt visitSideEffect(const T &op) {
        auto ret = BuiltinSimplify::visit(op);
        for (size_t i = 0, iEnd = iterStack_.size(); i < iEnd; i++) {
            auto &&var = iterStack_[i];
            auto &&defs = defStack_[i];
            auto hash = getHash(var);
            auto bound = transient(var);
            Expr lower, upper;
            for (auto &&first : bound.first) {
                if (checkAllDefined(defs, first)) {
                    lower = lower.isValid() ? makeMax(lower, first) : first;
                }
            }
            for (auto &&second : bound.second) {
                if (checkAllDefined(defs, second)) {
                    upper = upper.isValid() ? makeMin(upper, second) : second;
                }
            }
            // The bound can not be infinity, because it is a loop iterator
            ASSERT(lower.isValid() && upper.isValid());
            newRange_[hash].first = newRange_[hash].first.isValid()
                                        ? makeMin(newRange_[hash].first, lower)
                                        : lower;
            newRange_[hash].second =
                newRange_[hash].second.isValid()
                    ? makeMax(newRange_[hash].second, upper)
                    : upper;
        }
        return ret;
    }

  protected:
    using BuiltinSimplify::visit;

    Stmt visit(const Store &op) override { return visitSideEffect(op); }
    Stmt visit(const ReduceTo &op) override { return visitSideEffect(op); }
    // TODO: Also for Eval with side effect
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
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
