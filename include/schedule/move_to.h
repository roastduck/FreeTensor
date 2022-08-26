#ifndef FREE_TENSOR_MOVE_TO_H
#define FREE_TENSOR_MOVE_TO_H

#include <stmt.h>

namespace freetensor {

enum class MoveToSide : int { Before, After };
inline std::ostream &operator<<(std::ostream &os, MoveToSide side) {
    return os << (side == MoveToSide::Before ? "before" : "after");
}

std::pair<Stmt, std::pair<ID, ID>> moveTo(const Stmt &ast, const ID &stmt,
                                          MoveToSide side, const ID &dst);

} // namespace freetensor

#endif // FREE_TENSOR_MOVE_TO_H
