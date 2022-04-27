#ifndef FREE_TENSOR_REPLACE_ITER_H
#define FREE_TENSOR_REPLACE_ITER_H

#include <unordered_map>

#include <mutator.h>

namespace freetensor {

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

} // namespace freetensor

#endif // FREE_TENSOR_REPLACE_ITER_H
