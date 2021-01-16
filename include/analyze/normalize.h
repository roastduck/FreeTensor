#ifndef NORMALIZE_H
#define NORMALIZE_H

#include <mutator.h>

namespace ir {

/**
 * Hint all conditions, min/max and loops with normalized forms
 */
class Normalize : public Mutator {
  private:
    template <class T> Expr doNormalize(const T &_op) {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == _op->nodeType());
        auto op = __op.template as<typename T::Object>();
        op->info_norm_form_ = makeSub(op->lhs_, op->rhs_);
        return op;
    }

  protected:
    virtual Expr visit(const LT &op) override { return doNormalize(op); }
    virtual Expr visit(const LE &op) override { return doNormalize(op); }
    virtual Expr visit(const GT &op) override { return doNormalize(op); }
    virtual Expr visit(const GE &op) override { return doNormalize(op); }
    virtual Expr visit(const EQ &op) override { return doNormalize(op); }
    virtual Expr visit(const NE &op) override { return doNormalize(op); }
    virtual Expr visit(const Min &op) override { return doNormalize(op); }
    virtual Expr visit(const Max &op) override { return doNormalize(op); }
    virtual Stmt visit(const For &op) override;
};

inline Stmt normalize(const Stmt &op) { return Normalize()(op); }

} // namespace ir

#endif // NORMALIZE_H
