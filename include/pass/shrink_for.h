#ifndef SHRINK_FOR_H
#define SHRINK_FOR_H

#include <itertools.hpp>

#include <analyze/check_all_defined.h>
#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds.h>
#include <analyze/symbol_table.h>
#include <analyze/type_infer.h>
#include <func.h>
#include <hash.h>
#include <mutator.h>

namespace ir {

class ShrinkFor
    : public CompTransientBounds<WithTypeInfer<SymbolTable<Mutator>>> {
    typedef CompTransientBounds<WithTypeInfer<SymbolTable<Mutator>>> BaseClass;

    CompUniqueBounds bound_;

    ASTHashMap<Var, std::pair<std::vector<std::vector<Expr>>,
                              std::vector<std::vector<Expr>>>>
        newRange_;
    std::vector<Var> iterStack_;
    std::vector<std::unordered_set<std::string>> namesStack_;

  public:
    ShrinkFor() : bound_(*this, *this) {}

  private:
    template <class T> Stmt visitSideEffect(const T &op) {
        auto ret = BaseClass::visit(op);
        for (auto &&[var, names] : iter::zip(iterStack_, namesStack_)) {
            auto tr = transient(var);
            std::vector<Expr> lower, upper;
            for (auto &&first : tr.lower_) {
                if (checkAllDefined(names, first)) {
                    lower.emplace_back(first);
                } else {
                    for (auto &&l : bound_.getLower(first)) {
                        if (auto &&expr = l.expr();
                            checkAllDefined(names, expr)) {
                            lower.emplace_back(expr);
                        }
                    }
                }
            }
            for (auto &&second : tr.upper_) {
                if (checkAllDefined(names, second)) {
                    upper.emplace_back(second);
                } else {
                    for (auto &&u : bound_.getUpper(second)) {
                        if (auto &&expr = u.expr();
                            checkAllDefined(names, expr)) {
                            upper.emplace_back(expr);
                        }
                    }
                }
            }
            newRange_[var].first.emplace_back(std::move(lower));
            newRange_[var].second.emplace_back(std::move(upper));
        }
        return ret;
    }

  protected:
    using BaseClass::visit;

    Stmt visit(const Store &op) override { return visitSideEffect(op); }
    Stmt visit(const ReduceTo &op) override { return visitSideEffect(op); }
    // TODO: Also for Eval with side effect
    Stmt visit(const For &op) override;
};

/**
 * Increase the begin and decrease the end index, to remove redundant iterations
 * from For loops
 */
Stmt shrinkFor(const Stmt &op);

DEFINE_PASS_FOR_FUNC(shrinkFor)

} // namespace ir

#endif // SHRINK_FOR_H
