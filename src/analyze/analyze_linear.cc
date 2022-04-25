#include <analyze/analyze_linear.h>

namespace ir {

void AnalyzeLinear::visitExpr(const Expr &op) {
    if (!result_.count(op)) {
        Visitor::visitExpr(op);
        if (!result_.count(op)) {
            result_[op] = {{{1, op}}, 0};
        }
    }
}

void AnalyzeLinear::visit(const IntConst &op) {
    Visitor::visit(op);
    result_[op] = {{}, op->val_};
}

void AnalyzeLinear::visit(const Add &op) {
    Visitor::visit(op);
    const auto &e1 = result_.at(op->lhs_);
    const auto &e2 = result_.at(op->rhs_);
    result_[op] = add(e1, e2);
}

void AnalyzeLinear::visit(const Sub &op) {
    Visitor::visit(op);
    const auto &e1 = result_.at(op->lhs_);
    const auto &e2 = result_.at(op->rhs_);
    result_[op] = sub(e1, e2);
}

void AnalyzeLinear::visit(const Mul &op) {
    Visitor::visit(op);
    const auto &e1 = result_.at(op->lhs_);
    const auto &e2 = result_.at(op->rhs_);

    if (e1.isConst()) {
        result_[op] = mul(e2, e1.bias_);
        return;
    }
    if (e2.isConst()) {
        result_[op] = mul(e1, e2.bias_);
        return;
    }
    // Not linear
}

LinearExpr<int64_t> linear(const Expr &expr) {
    AnalyzeLinear visitor;
    visitor(expr);
    return visitor.result().at(expr);
}

Opt<std::pair<LinearExpr<int64_t>, ASTNodeType>> linearComp(const Expr &expr) {
    switch (expr->nodeType()) {
    case ASTNodeType::LT:
    case ASTNodeType::LE:
    case ASTNodeType::GT:
    case ASTNodeType::GE:
    case ASTNodeType::EQ:
    case ASTNodeType::NE:
        return Opt<std::pair<LinearExpr<int64_t>, ASTNodeType>>::make(
            std::make_pair(linear(makeSub(expr.as<BinaryExprNode>()->lhs_,
                                          expr.as<BinaryExprNode>()->rhs_)),
                           expr->nodeType()));
    default:
        return nullptr;
    }
}

} // namespace ir
