#include <pass/replace_uses.h>

namespace ir {

Expr ReplaceUses::visit(const Load &op) {
    if (replaceLoad_.count(op)) {
        return (*this)(replaceLoad_.at(op));
    } else {
        return Mutator::visit(op);
    }
}

} // namespace ir

