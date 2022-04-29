#include <analyze/all_uses.h>

namespace freetensor {

void AllUses::visit(const Load &op) {
    if (!noRecurseIdx_) {
        Visitor::visit(op);
    }
    if (type_ & CHECK_LOAD) {
        uses_.insert(op->var_);
    }
}

void AllUses::visit(const Store &op) {
    if (!noRecurseIdx_) {
        Visitor::visit(op);
    }
    if (type_ & CHECK_STORE) {
        uses_.insert(op->var_);
    }
}

void AllUses::visit(const ReduceTo &op) {
    if (!noRecurseIdx_) {
        Visitor::visit(op);
    }
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

std::unordered_set<std::string>
allUses(const AST &op, AllUses::AllUsesType type, bool noRecurseIdx) {
    AllUses visitor(type, noRecurseIdx);
    visitor(op);
    return visitor.uses();
}

std::unordered_set<std::string> allReads(const AST &op, bool noRecurseIdx) {
    return allUses(op, AllUses::CHECK_LOAD, noRecurseIdx);
}

std::unordered_set<std::string> allWrites(const AST &op, bool noRecurseIdx) {
    return allUses(op, AllUses::CHECK_STORE | AllUses::CHECK_REDUCE,
                   noRecurseIdx);
}

std::unordered_set<std::string> allIters(const AST &op, bool noRecurseIdx) {
    return allUses(op, AllUses::CHECK_VAR, noRecurseIdx);
}

std::unordered_set<std::string> allNames(const AST &op, bool noRecurseIdx) {
    return allUses(op,
                   AllUses::CHECK_LOAD | AllUses::CHECK_STORE |
                       AllUses::CHECK_REDUCE | AllUses::CHECK_VAR,
                   noRecurseIdx);
}

} // namespace freetensor
