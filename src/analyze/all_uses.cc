#include <analyze/all_uses.h>

namespace ir {

void AllUses::visit(const Load &op) {
    Visitor::visit(op);
    if (type_ & CHECK_LOAD) {
        uses_.insert(op->var_);
    }
}

void AllUses::visit(const Store &op) {
    Visitor::visit(op);
    if (type_ & CHECK_STORE) {
        uses_.insert(op->var_);
    }
}

void AllUses::visit(const ReduceTo &op) {
    Visitor::visit(op);
    if (type_ & CHECK_REDUCE) {
        uses_.insert(op->var_);
    }
}

void AllUses::visit(const Var &op) {
    Visitor::visit(op);
    if (type_ & CHECK_VAR) {
        uses_.insert(op->name_);
    }
}

std::unordered_set<std::string> allUses(const AST &op,
                                        AllUses::AllUsesType type) {
    AllUses visitor(type);
    visitor(op);
    return visitor.uses();
}

std::unordered_set<std::string> allReads(const AST &op) {
    return allUses(op, AllUses::CHECK_LOAD);
}

std::unordered_set<std::string> allWrites(const AST &op) {
    return allUses(op, AllUses::CHECK_STORE | AllUses::CHECK_REDUCE);
}

std::unordered_set<std::string> allIters(const AST &op) {
    return allUses(op, AllUses::CHECK_VAR);
}

std::unordered_set<std::string> allNames(const AST &op) {
    return allUses(op, AllUses::CHECK_LOAD | AllUses::CHECK_STORE |
                           AllUses::CHECK_REDUCE | AllUses::CHECK_VAR);
}

} // namespace ir
