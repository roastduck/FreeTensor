#ifndef FREE_TENSOR_ALL_SIDE_EFFECT_INTRINSICS_H
#define FREE_TENSOR_ALL_SIDE_EFFECT_INTRINSICS_H

#include <unordered_set>

#include <visitor.h>

namespace freetensor {

/**
 * Record all intrinsics with side effects in an AST
 */
class FindSideEffectIntrinsics : public Visitor {
  public:
    std::unordered_set<Intrinsic> sideEffectIntrinsics_;

  protected:
    void visit(const Intrinsic &op) override {
        if (op->hasSideEffect_)
            sideEffectIntrinsics_.insert(op);
        Visitor::visit(op);
    }
};

template <typename T>
std::unordered_set<Intrinsic> allSideEffectIntrinsics(T &&op) {
    FindSideEffectIntrinsics finder;
    finder(op);
    return std::move(finder.sideEffectIntrinsics_);
}

} // namespace freetensor

#endif
