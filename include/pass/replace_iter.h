#ifndef REPLACE_ITER_H
#define REPLACE_ITER_H

#include <mutator.h>

namespace ir {

/**
 * Replace all Var node with a specific name by another expression
 */
class ReplaceIter : public Mutator {
    std::string name_;
    Expr expr_;

  public:
    ReplaceIter(const std::string &name, const Expr &expr)
        : name_(name), expr_(expr) {}

  protected:
    Expr visit(const Var &op) override {
        if (op->name_ == name_) {
            return (*this)(expr_);
        } else {
            return Mutator::visit(op);
        }
    }
};

} // namespace ir

#endif // REPLACE_ITER_H
