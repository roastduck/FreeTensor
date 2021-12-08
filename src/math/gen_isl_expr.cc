#include <math/gen_isl_expr.h>
#include <math/utils.h>

namespace ir {

std::string GenISLExpr::normalizeId(const std::string &old) {
    if (idCache_.count(old)) {
        return idCache_.at(old);
    }
    std::string ret = old;
    for (char &c : ret) {
        if (!isalnum(c) && c != '_') {
            c = '_';
        }
    }
    while (idFlag_.count(ret)) {
        ret += "_";
    }
    idFlag_.insert(ret);
    return idCache_[old] = ret;
}

void GenISLExpr::visitExpr(const Expr &op) {
    if (!visited_.count(op)) {
        Visitor::visitExpr(op);
        visited_.insert(op);
    }
}

void GenISLExpr::visit(const Var &op) { results_[op] = normalizeId(op->name_); }

void GenISLExpr::visit(const IntConst &op) {
    results_[op] = std::to_string(op->val_);
    constants_[op] = op->val_;
}

void GenISLExpr::visit(const Add &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " + " + results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                std::to_string(constants_[op] = constants_.at(op->lhs_) +
                                                constants_.at(op->rhs_));
        }
    }
}

void GenISLExpr::visit(const Sub &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " - " + results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                std::to_string(constants_[op] = constants_.at(op->lhs_) -
                                                constants_.at(op->rhs_));
        }
    }
}

void GenISLExpr::visit(const Mul &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        if (constants_.count(op->lhs_) || constants_.count(op->rhs_)) {
            results_[op] = "(" + results_.at(op->lhs_) + " * " +
                           results_.at(op->rhs_) + ")";
        }
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                std::to_string(constants_[op] = constants_.at(op->lhs_) *
                                                constants_.at(op->rhs_));
        }
    }
}

void GenISLExpr::visit(const LAnd &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " and " + results_.at(op->rhs_) + ")";
    }
}

void GenISLExpr::visit(const LOr &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " or " + results_.at(op->rhs_) + ")";
    }
}

void GenISLExpr::visit(const LNot &op) {
    Visitor::visit(op);
    if (results_.count(op->expr_)) {
        results_[op] = "(not " + results_.at(op->expr_) + ")";
    }
}

void GenISLExpr::visit(const LT &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " < " + results_.at(op->rhs_);
    }
}

void GenISLExpr::visit(const LE &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " <= " + results_.at(op->rhs_);
    }
}

void GenISLExpr::visit(const GT &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " > " + results_.at(op->rhs_);
    }
}

void GenISLExpr::visit(const GE &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " >= " + results_.at(op->rhs_);
    }
}

void GenISLExpr::visit(const EQ &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " = " + results_.at(op->rhs_);
    }
}

void GenISLExpr::visit(const NE &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " != " + results_.at(op->rhs_);
    }
}

void GenISLExpr::visit(const FloorDiv &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && constants_.count(op->rhs_)) {
        results_[op] = "floor(" + results_.at(op->lhs_) + " / " +
                       results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] = std::to_string(
                constants_[op] =
                    floorDiv(constants_.at(op->lhs_), constants_.at(op->rhs_)));
        }
    }
}

void GenISLExpr::visit(const CeilDiv &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && constants_.count(op->rhs_)) {
        results_[op] = "ceil(" + results_.at(op->lhs_) + " / " +
                       results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] = std::to_string(
                constants_[op] =
                    ceilDiv(constants_.at(op->lhs_), constants_.at(op->rhs_)));
        }
    }
}

void GenISLExpr::visit(const Mod &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && constants_.count(op->rhs_)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " % " + results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                std::to_string(constants_[op] = constants_.at(op->lhs_) %
                                                constants_.at(op->rhs_));
        }
    }
}

void GenISLExpr::visit(const Min &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "min(" + results_.at(op->lhs_) + ", " + results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] = std::to_string(
                constants_[op] =
                    std::min(constants_.at(op->lhs_), constants_.at(op->rhs_)));
        }
    }
}

void GenISLExpr::visit(const Max &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "max(" + results_.at(op->lhs_) + ", " + results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] = std::to_string(
                constants_[op] =
                    std::max(constants_.at(op->lhs_), constants_.at(op->rhs_)));
        }
    }
}

Ref<std::string> GenISLExpr::gen(const Expr &op) {
    (*this)(op);
    if (results_.count(op)) {
        return Ref<std::string>::make(results_.at(op));
    }
    return nullptr;
}

} // namespace ir

