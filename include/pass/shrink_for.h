#ifndef SHRINK_FOR_H
#define SHRINK_FOR_H

#include <analyze/check_all_defined.h>
#include <func.h>
#include <pass/simplify.h>

namespace ir {

class ShrinkFor : public CompUniqueBounds {
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
            auto tr = transient(var);
            std::vector<Expr> lower, upper;
            for (auto &&first : tr.lower_) {
                if (checkAllDefined(defs, first)) {
                    lower.emplace_back(first);
                } else {
                    for (auto &&l : getLower((*this)(first))) {
                        if (auto &&expr = l.expr();
                            checkAllDefined(defs, expr)) {
                            lower.emplace_back(expr);
                        }
                    }
                }
            }
            for (auto &&second : tr.upper_) {
                if (checkAllDefined(defs, second)) {
                    upper.emplace_back(second);
                } else {
                    for (auto &&u : getUpper((*this)(second))) {
                        if (auto &&expr = u.expr();
                            checkAllDefined(defs, expr)) {
                            upper.emplace_back(expr);
                        }
                    }
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

DEFINE_PASS_FOR_FUNC(shrinkFor)

} // namespace ir

#endif // SHRINK_FOR_H
