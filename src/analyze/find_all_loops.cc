#include <analyze/find_all_loops.h>

namespace ir {

void FindAllLoops::visit(const For &op) {
    Visitor::visit(op);
    loops_.emplace_back(op->id());
}

std::vector<ID> findAllLoops(const Stmt &op) {
    FindAllLoops visitor;
    visitor(op);
    return visitor.loops();
}

} // namespace ir

