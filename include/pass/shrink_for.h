#ifndef SHRINK_FOR_H
#define SHRINK_FOR_H

#include <analyze/check_all_defined.h>
#include <func.h>
#include <pass/simplify.h>

namespace ir {

class ShrinkFor : public CompTransientBounds {
    std::unordered_map<uint64_t, std::pair<std::vector<std::vector<Expr>>,
                                           std::vector<std::vector<Expr>>>>
        newRange_;
    std::vector<Var> iterStack_;
    std::vector<std::unordered_set<std::string>> defStack_;
    std::unordered_set<std::string> defs_;

  private:
    template <class T> Stmt visitSideEffect(const T &op) {
        auto ret = CompTransientBounds::visit(op);
        for (size_t i = 0, iEnd = iterStack_.size(); i < iEnd; i++) {
            auto &&var = iterStack_[i];
            auto &&defs = defStack_[i];
            auto hash = getHash(var);
            auto bound = transient(var);
            std::vector<Expr> lower, upper;
            for (auto &&first : bound.lower_) {
                if (checkAllDefined(defs, first)) {
                    lower.emplace_back(first);
                }
            }
            for (auto &&second : bound.upper_) {
                if (checkAllDefined(defs, second)) {
                    upper.emplace_back(second);
                }
            }
            newRange_[hash].first.emplace_back(std::move(lower));
            newRange_[hash].second.emplace_back(std::move(upper));
        }
        return ret;
    }

  protected:
    using CompTransientBounds::visit;

    Stmt visit(const Store &op) override { return visitSideEffect(op); }
    Stmt visit(const ReduceTo &op) override { return visitSideEffect(op); }
    // TODO: Also for Eval with side effect
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
};

/**
 * Increase the begin and decrease the end index, to remove redundant iterations
 * from For loops
 */
Stmt shrinkFor(const Stmt &op);

inline Func shrinkFor(const Func &func) {
    return makeFunc(func->name_, func->params_, shrinkFor(func->body_),
                    func->src_);
}

} // namespace ir

#endif // SHRINK_FOR_H
