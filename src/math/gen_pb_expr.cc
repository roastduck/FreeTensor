#include <math/gen_pb_expr.h>
#include <math/utils.h>
#include <serialize/mangle.h>
#include <serialize/print_ast.h>

namespace freetensor {

template <class T, class V, class Hash, class KeyEqual>
static void unionTo(std::unordered_map<T, V, Hash, KeyEqual> &target,
                    const std::unordered_map<T, V, Hash, KeyEqual> &other) {
    target.insert(other.begin(), other.end());
}

static std::string boolToStr(bool v) { return v ? "true" : "false"; }

void GenPBExpr::visitExpr(const Expr &op) {
    if (!results_.count(op)) {
        auto oldParent = parent_;
        parent_ = op;
        Visitor::visitExpr(op);
        parent_ = oldParent;

        if (!results_.count(op)) {
            bool noNeedToBeVar = false;
            if (noNeedToBeVars_.count(op)) {
                // Our caller explicitly mark this sub-expression as no need to
                // be a dedicated free variable
                noNeedToBeVar = true;
            }
            if (!isBool(op->dtype()) && !isInt(op->dtype())) {
                // Data types like float are certainly no need to be a dedicated
                // presburger variable
                noNeedToBeVar = true;
            }
            if (!noNeedToBeVar || !parent_.isValid()) {
                auto freeVar =
                    mangle(dumpAST(op, true)) + "__ext__" + varSuffix_;

                // Since the expression is already a free variable, no need
                // to make other free variables for is sub-expressions
                vars_[op] = {};

                if (op->dtype() == DataType::Bool) {
                    // Treat the free variable as an integer because ISL
                    // does not support bool variables. NOTE: When we are
                    // parsing ISL objects back to AST in math/parse_pb_expr, we
                    // need to recover bool variables.
                    results_[op] = "(" + freeVar + " > 0)";
                } else {
                    results_[op] = freeVar;
                }
                vars_[op][op] = freeVar;
            }
        }
    }

    if (parent_.isValid()) {
        unionTo(vars_[parent_], vars_[op]);
    }
}

void GenPBExpr::visit(const Var &op) {
    auto str = mangle(op->name_);
    vars_[op][op] = str;
    results_[op] = str;
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
    if (op->lhs_->dtype() == DataType::Bool ||
        op->rhs_->dtype() == DataType::Bool) {
        return; // TODO: Convert to a boolean expression
    }
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
    if (op->lhs_->dtype() == DataType::Bool ||
        op->rhs_->dtype() == DataType::Bool) {
        return; // TODO: Convert to a boolean expression
    }
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
    // ISL requires a positive divisor
    if (results_.count(op->lhs_) && constants_.count(op->rhs_)) {
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
    // ISL requires a positive divisor
    if (results_.count(op->lhs_) && constants_.count(op->rhs_)) {
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
    // ISL requires a positive divisor
    if (results_.count(op->lhs_) && constants_.count(op->rhs_)) {
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
    if (constants_.count(op->cond_)) {
        if (constants_.at(op->cond_)) {
            if (results_.count(op->thenCase_)) {
                results_[op] = results_.at(op->thenCase_);
                if (constants_.count(op->thenCase_)) {
                    constants_[op] = constants_.at(op->thenCase_);
                }
            }
        } else {
            if (results_.count(op->elseCase_)) {
                results_[op] = results_.at(op->elseCase_);
                if (constants_.count(op->elseCase_)) {
                    constants_[op] = constants_.at(op->elseCase_);
                }
            }
        }
    }
}

void GenPBExpr::visit(const Unbound &op) {
    Visitor::visit(op);
    // Just ignore this node
    if (auto it = results_.find(op->expr_); it != results_.end()) {
        results_[op] = it->second;
    }
    if (auto it = constants_.find(op->expr_); it != constants_.end()) {
        constants_[op] = it->second;
    }
    if (auto it = vars_.find(op->expr_); it != vars_.end()) {
        vars_[op] = it->second;
    }
}

std::pair<std::string, GenPBExpr::VarMap> GenPBExpr::gen(const Expr &op) {
    (*this)(op);
    if (vars_.count(op)) {
        return {results_.at(op), vars_.at(op)};
    } else {
        return {results_.at(op), {}};
    }
}

} // namespace freetensor
