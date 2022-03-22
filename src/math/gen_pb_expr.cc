#include <mangle.h>
#include <math/gen_pb_expr.h>
#include <math/utils.h>

namespace ir {

template <class T, class V, class Hash, class KeyEqual>
static void unionTo(std::unordered_map<T, V, Hash, KeyEqual> &target,
                    const std::unordered_map<T, V, Hash, KeyEqual> &other) {
    target.insert(other.begin(), other.end());
}

static std::string boolToStr(bool v) { return v ? "true" : "false"; }

void GenPBExpr::visitExpr(const Expr &op) {
    auto oldParent = parent_;
    parent_ = op;

    if (!visited_.count(op)) {
        Visitor::visitExpr(op);
        visited_.insert(op);
    }

    parent_ = oldParent;
    if (parent_.isValid()) {
        unionTo(vars_[parent_], vars_[op]);
    }
}

void GenPBExpr::visit(const Var &op) {
    auto str = mangle(op->name_);
    vars_[op][op] = str;
    results_[op] = str;
}

void GenPBExpr::visit(const Load &op) {
    if (isInt(symbolTable_.buffer(op->var_)->tensor().dtype())) {
        auto str = mangle("ext" + std::to_string(op->hash())) + varSuffix_;
        vars_[op][op] = str;
        results_[op] = str;
    }
}

void GenPBExpr::visit(const IntConst &op) {
    results_[op] = std::to_string(op->val_);
    constants_[op] = op->val_;
}

void GenPBExpr::visit(const BoolConst &op) {
    results_[op] = boolToStr(op->val_);
    constants_[op] = op->val_;
}

void GenPBExpr::visit(const Add &op) {
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

void GenPBExpr::visit(const Sub &op) {
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

void GenPBExpr::visit(const Mul &op) {
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

void GenPBExpr::visit(const LAnd &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " and " + results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                boolToStr((constants_[op] = (constants_.at(op->lhs_) &&
                                             constants_.at(op->rhs_))));
        }
    }
}

void GenPBExpr::visit(const LOr &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " or " + results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                boolToStr((constants_[op] = (constants_.at(op->lhs_) ||
                                             constants_.at(op->rhs_))));
        }
    }
}

void GenPBExpr::visit(const LNot &op) {
    Visitor::visit(op);
    if (results_.count(op->expr_)) {
        results_[op] = "(not " + results_.at(op->expr_) + ")";
        if (constants_.count(op->expr_)) {
            results_[op] =
                boolToStr((constants_[op] = !constants_.at(op->expr_)));
        }
    }
}

void GenPBExpr::visit(const LT &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " < " + results_.at(op->rhs_);
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                boolToStr((constants_[op] = (constants_.at(op->lhs_) <
                                             constants_.at(op->rhs_))));
        }
    }
}

void GenPBExpr::visit(const LE &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " <= " + results_.at(op->rhs_);
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                boolToStr((constants_[op] = (constants_.at(op->lhs_) <=
                                             constants_.at(op->rhs_))));
        }
    }
}

void GenPBExpr::visit(const GT &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " > " + results_.at(op->rhs_);
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                boolToStr((constants_[op] = (constants_.at(op->lhs_) >
                                             constants_.at(op->rhs_))));
        }
    }
}

void GenPBExpr::visit(const GE &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " >= " + results_.at(op->rhs_);
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                boolToStr((constants_[op] = (constants_.at(op->lhs_) >=
                                             constants_.at(op->rhs_))));
        }
    }
}

void GenPBExpr::visit(const EQ &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " = " + results_.at(op->rhs_);
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                boolToStr((constants_[op] = (constants_.at(op->lhs_) ==
                                             constants_.at(op->rhs_))));
        }
    }
}

void GenPBExpr::visit(const NE &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " != " + results_.at(op->rhs_);
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                boolToStr((constants_[op] = (constants_.at(op->lhs_) !=
                                             constants_.at(op->rhs_))));
        }
    }
}

void GenPBExpr::visit(const FloorDiv &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && constants_.count(op->rhs_)) {
        // ISL requires a positive divisor
        if (constants_.at(op->rhs_) > 0) {
            results_[op] = "floor(" + results_.at(op->lhs_) + " / " +
                           std::to_string(constants_.at(op->rhs_)) + ")";
        } else {
            results_[op] = "floor(-(" + results_.at(op->lhs_) + ") / " +
                           std::to_string(-constants_.at(op->rhs_)) + ")";
        }
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] = std::to_string(
                constants_[op] =
                    floorDiv(constants_.at(op->lhs_), constants_.at(op->rhs_)));
        }
    }
}

void GenPBExpr::visit(const CeilDiv &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && constants_.count(op->rhs_)) {
        // ISL requires a positive divisor
        if (constants_.at(op->rhs_) > 0) {
            results_[op] = "ceil(" + results_.at(op->lhs_) + " / " +
                           std::to_string(constants_.at(op->rhs_)) + ")";
        } else {
            results_[op] = "ceil(-(" + results_.at(op->lhs_) + ") / " +
                           std::to_string(-constants_.at(op->rhs_)) + ")";
        }
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] = std::to_string(
                constants_[op] =
                    ceilDiv(constants_.at(op->lhs_), constants_.at(op->rhs_)));
        }
    }
}

void GenPBExpr::visit(const Mod &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && constants_.count(op->rhs_)) {
        // ISL requires a positive divisor
        if (constants_.at(op->rhs_) > 0) {
            results_[op] = "(" + results_.at(op->lhs_) + " % " +
                           std::to_string(constants_.at(op->rhs_)) + ")";
        } else {
            results_[op] = "(-(" + results_.at(op->lhs_) + ") % " +
                           std::to_string(-constants_.at(op->rhs_)) + ")";
        }
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                std::to_string(constants_[op] = constants_.at(op->lhs_) %
                                                constants_.at(op->rhs_));
        }
    }
}

void GenPBExpr::visit(const Min &op) {
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

void GenPBExpr::visit(const Max &op) {
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

void GenPBExpr::visit(const IfExpr &op) {
    Visitor::visit(op);
    if (constants_.count(op->cond_) && results_.count(op->thenCase_) &&
        results_.count(op->elseCase_)) {
        if (constants_.at(op->cond_)) {
            results_[op] = results_.at(op->thenCase_);
            if (constants_.count(op->thenCase_)) {
                constants_[op] = constants_.at(op->thenCase_);
            }
        } else {
            results_[op] = results_.at(op->elseCase_);
            if (constants_.count(op->elseCase_)) {
                constants_[op] = constants_.at(op->elseCase_);
            }
        }
    }
}

Opt<std::string> GenPBExpr::gen(const Expr &op) {
    (*this)(op);
    if (results_.count(op)) {
        return Opt<std::string>::make(results_.at(op));
    }
    return nullptr;
}

} // namespace ir
