#ifndef DISAMBIGUOUS_H
#define DISAMBIGUOUS_H

#include <mutator.h>

namespace ir {

/**
 * Make there will be no shared node in a AST
 *
 * If any pass uses node addresses as keys, perform Disambiguous first
 */
class Disambiguous : public Mutator {
  public:
    Stmt operator()(const Stmt &op) override {
        if (op->noAmbiguous_) {
            return op;
        }
        auto ret = Mutator::operator()(op);
        ret->noAmbiguous_ = true;
        return ret;
    }

    Expr operator()(const Expr &op) override {
        if (op->noAmbiguous_) {
            return op;
        }
        auto ret = Mutator::operator()(op);
        ret->noAmbiguous_ = true;
        return ret;
    }
};

} // namespace ir

#endif // DISAMBIGUOUS_H
