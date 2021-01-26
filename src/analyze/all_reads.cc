#include <analyze/all_reads.h>

namespace ir {

void AllReads::visit(const Load &op) {
    Visitor::visit(op);
    reads_.insert(op->var_);
}

std::unordered_set<std::string> allReads(const AST &op) {
    AllReads visitor;
    visitor(op);
    return visitor.reads();
}

} // namespace ir

