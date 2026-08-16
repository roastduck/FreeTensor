#ifndef FREE_TENSOR_MOVE_OUT_FIRST_OR_LAST_ITER_H
#define FREE_TENSOR_MOVE_OUT_FIRST_OR_LAST_ITER_H

#include <unordered_map>

#include <func.h>
#include <pass/z3_simplify.h>
#include <subprocess.h>

namespace freetensor {

class MoveOutFirstOrLastIter : public Z3SimplifyWithSymbolTable {
  protected:
    Stmt visit(const For &op) override;
};

/**
 * Transform loop with special case for the first or the last iteration into a
 * loop without these cases
 *
 * E.g.
 *
 * Transform
 *
 * for i = 0 to n
 *   if i == 0
 *     A
 *   B
 *
 * to be
 *
 * A
 * for i = 0 to n
 *   B
 */
inline Stmt
moveOutFirstOrLastIter(const Stmt &op,
                       const std::optional<bool> &asSubprocess = std::nullopt,
                       const std::optional<double> &timeout = std::nullopt) {
    if (shouldRunInSubprocess(asSubprocess, timeout)) {
        return runPassSubprocess<Stmt>("move_out_first_or_last_iter", op, {},
                                       asSubprocess, timeout);
    }
    return MoveOutFirstOrLastIter()(op);
}

DEFINE_PASS_FOR_FUNC(moveOutFirstOrLastIter)

} // namespace freetensor

#endif // FREE_TENSOR_MOVE_OUT_FIRST_OR_LAST_ITER_H
