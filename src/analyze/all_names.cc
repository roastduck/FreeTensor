#include <analyze/all_names.h>

namespace ir {

void AllNames::visit(const Var &op) {
    Visitor::visit(op);
    names_.insert(op->name_);
}

void AllNames::visit(const Load &op) {
    Visitor::visit(op);
    names_.insert(op->var_);
}

void AllNames::visit(const Store &op) {
    Visitor::visit(op);
    names_.insert(op->var_);
}

void AllNames::visit(const ReduceTo &op) {
    Visitor::visit(op);
    names_.insert(op->var_);
}

std::unordered_set<std::string> allNames(const AST &op) {
    AllNames visitor;
    visitor(op);
    return visitor.names();
}

} // namespace ir
