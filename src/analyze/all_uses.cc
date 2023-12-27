#include <analyze/all_uses.h>

namespace freetensor {

void AllUses::visitStmt(const Stmt &s) {
    if (!inFirstStmt_ || !noRecurseSubStmt_) {
        bool oldInFirstStmt = inFirstStmt_;
        inFirstStmt_ = true;
        Visitor::visitStmt(s);
        inFirstStmt_ = oldInFirstStmt;
    }
}

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
    } else {
        (*this)(op->expr_);
    }
    if (type_ & CHECK_STORE) {
        uses_.insert(op->var_);
    }
}

void AllUses::visit(const ReduceTo &op) {
    if (!noRecurseIdx_) {
        Visitor::visit(op);
    } else {
        (*this)(op->expr_);
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

std::unordered_set<std::string> allUses(const AST &op,
                                        AllUses::AllUsesType type,
                                        bool noRecurseIdx,
                                        bool noRecurseSubStmt) {
    AllUses visitor(type, noRecurseIdx, noRecurseSubStmt);
    visitor(op);
    return visitor.uses();
}

} // namespace freetensor
