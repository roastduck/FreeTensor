#include <analyze/check_all_defined.h>

namespace ir {

void CheckAllDefined::visit(const Var &op) {
    if (!allDef_) {
        return;
    }
    Visitor::visit(op);
    allDef_ &= !!defs_.count(op->name_);
}

void CheckAllDefined::visit(const Load &op) {
    if (!allDef_) {
        return;
    }
    Visitor::visit(op);
    allDef_ &= !!defs_.count(op->var_);
}

void CheckAllDefined::visit(const Store &op) {
    if (!allDef_) {
        return;
    }
    Visitor::visit(op);
    allDef_ &= !!defs_.count(op->var_);
}

void CheckAllDefined::visit(const ReduceTo &op) {
    if (!allDef_) {
        return;
    }
    Visitor::visit(op);
    allDef_ &= !!defs_.count(op->var_);
}

bool checkAllDefined(const std::unordered_set<std::string> &defs,
                    const AST &op) {
    CheckAllDefined visitor(defs);
    visitor(op);
    return visitor.allDef();
}

} // namespace ir

