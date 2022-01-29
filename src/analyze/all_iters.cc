#include <analyze/all_iters.h>

namespace ir {

void AllIters::visit(const Var &op) {
    Visitor::visit(op);
    iters_.insert(op->name_);
}

std::unordered_set<std::string> allIters(const AST &op) {
    AllIters visitor;
    visitor(op);
    return visitor.iters();
}

} // namespace ir

