#ifndef FREE_TENSOR_MAKE_REDUCTION_H
#define FREE_TENSOR_MAKE_REDUCTION_H

#include <unordered_set>

#include <func.h>
#include <mutator.h>

namespace freetensor {

class MakeReduction : public Mutator {
    const std::unordered_set<ReduceOp> &types_;
    bool canonicalOnly_;

  public:
    MakeReduction(const std::unordered_set<ReduceOp> &types, bool canonicalOnly)
        : types_(types), canonicalOnly_(canonicalOnly) {}

  private:
    bool isSameElem(const Store &s, const Load &l);

    Stmt doMake(Store op, ReduceOp reduceOp);

  protected:
    Stmt visit(const Store &op) override;
};

/**
 * Transform things like a = a + b into a += b
 *
 * This is to make the dependency analysis more accurate
 *
 * @param types : Only transform these types of reductions
 * @param canonicalOnly : True to avoid cyclic reductions like a += a + b
 */
inline Stmt makeReduction(const Stmt &op,
                          const std::unordered_set<ReduceOp> &types,
                          bool canonicalOnly = false) {
    return MakeReduction(types, canonicalOnly)(op);
}

inline Stmt makeReduction(const Stmt &op) {
    return makeReduction(op, {ReduceOp::Add, ReduceOp::Mul, ReduceOp::Min,
                              ReduceOp::Max, ReduceOp::LAnd, ReduceOp::LOr});
}

DEFINE_PASS_FOR_FUNC(makeReduction)

} // namespace freetensor

#endif // FREE_TENSOR_MAKE_REDUCTION_H
