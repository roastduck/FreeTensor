#ifndef MOVE_OUT_FIRST_OR_LAST_ITER_H
#define MOVE_OUT_FIRST_OR_LAST_ITER_H

#include <unordered_map>

#include <func.h>
#include <pass/z3_simplify.h>

namespace ir {

class MoveOutFirstOrLastIter : public Z3Simplify {
    std::unordered_map<std::string, Expr> replace_;

  protected:
    Expr visit(const Var &op) override;
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
inline Stmt moveOutFirstOrLastIter(const Stmt &op) {
    return MoveOutFirstOrLastIter()(op);
}

DEFINE_PASS_FOR_FUNC(moveOutFirstOrLastIter)

} // namespace ir

#endif // MOVE_OUT_FIRST_OR_LAST_ITER_H
