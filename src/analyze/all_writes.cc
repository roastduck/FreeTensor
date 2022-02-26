#include <analyze/all_writes.h>

namespace ir {

void AllWrites::visit(const Store &op) {
    Visitor::visit(op);
    writes_.insert(op->var_);
}

void AllWrites::visit(const ReduceTo &op) {
    Visitor::visit(op);
    writes_.insert(op->var_);
}

std::unordered_set<std::string> allWrites(const AST &op) {
    AllWrites visitor;
    visitor(op);
    return visitor.writes();
}

} // namespace ir
