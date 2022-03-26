#ifndef REPLACE_ITER_H
#define REPLACE_ITER_H

#include <unordered_map>

#include <mutator.h>

namespace ir {

/**
 * Replace all Var node with a specific name by another expression
 */
class ReplaceIter : public Mutator {
    std::unordered_map<std::string, Expr> replace_;

  public:
    ReplaceIter(const std::string &name, const Expr &expr)
        : replace_({{name, expr}}) {}
    ReplaceIter(const std::unordered_map<std::string, Expr> &replace)
        : replace_(replace) {}

  protected:
    Expr visit(const Var &op) override {
        if (replace_.count(op->name_)) {
            return (*this)(replace_.at(op->name_));
        } else {
            return Mutator::visit(op);
        }
    }
};

} // namespace ir

#endif // REPLACE_ITER_H
