#include <debug/print_ast.h>

#include "../codegen/detail/code_gen.h"

namespace ir {

constexpr const char *MAGENTA = "\u001b[35;1m";
constexpr const char *CYAN = "\u001b[36m";
constexpr const char *RESET = "\u001b[0m";
constexpr const char *BOLD = "\u001b[1m";

void PrintVisitor::recur(const Expr &op) {
    if (op.isValid()) {
        (*this)(op);
    } else {
        os() << "<NULL>";
    }
}

void PrintVisitor::recur(const Stmt &op) {
    if (op.isValid()) {
        (*this)(op);
    } else {
        makeIndent();
        os() << "<NULL>" << std::endl;
    }
}

void PrintVisitor::printId(const Stmt &op) {
#ifdef IR_DEBUG
    makeIndent();
    os() << "// By " << op->debugCreator_ << std::endl;
#endif
    if (op->hasNamedId()) {
        if (pretty_) {
            os() << CYAN << op->id() << ":" << RESET << std::endl;
        } else {
            os() << op->id() << ":" << std::endl;
        }
    }
}

void PrintVisitor::visitStmt(
    const Stmt &op, const std::function<void(const Stmt &)> &visitNode) {
    if (op->nodeType() != ASTNodeType::Any) {
        printId(op);
    }
    Visitor::visitStmt(op, visitNode);
}

void PrintVisitor::visit(const Func &op) {
    makeIndent();
    os() << "func(";
    for (size_t i = 0, iEnd = op->params_.size(); i < iEnd; i++) {
        os() << (i > 0 ? ", " : "") << op->params_[i];
    }
    os() << ") ";
    beginBlock();
    recur(op->body_);
    endBlock();
}

void PrintVisitor::visit(const Any &op) {
    makeIndent();
    os() << "<Any>" << std::endl;
}

void PrintVisitor::visit(const AnyExpr &op) { os() << "<Any>"; }

void PrintVisitor::visit(const VarDef &op) {
    makeIndent();
    os() << ::ir::toString(op->buffer_->atype()) << " "
         << ::ir::toString(op->buffer_->mtype()) << " " << op->name_ << ": ";
    auto &&tensor = op->buffer_->tensor();
    os() << ::ir::toString(tensor.dtype()) << "[";
    printList(tensor.shape());
    os() << "] ";
    if (op->sizeLim_.isValid()) {
        os() << "size_lim = ";
        recur(op->sizeLim_);
        os() << " ";
    }
    beginBlock();
    recur(op->body_);
    endBlock();
}

void PrintVisitor::visit(const Var &op) {
    os() << op->name_;
    Visitor::visit(op);
}

void PrintVisitor::visit(const Store &op) {
    makeIndent();
    os() << op->var_ << "[";
    printList(op->indices_);
    os() << "] = ";
    recur(op->expr_);
    os() << std::endl;
}

void PrintVisitor::visit(const Load &op) {
    os() << op->var_ << "[";
    printList(op->indices_);
    os() << "]";
}

void PrintVisitor::visit(const ReduceTo &op) {
    if (op->atomic_) {
        makeIndent();
        os() << "// atomic" << std::endl;
    }
    makeIndent();
    os() << op->var_ << "[";
    printList(op->indices_);
    os() << "] ";
    switch (op->op_) {
    case ReduceOp::Add:
        os() << "+=";
        break;
    case ReduceOp::Mul:
        os() << "*=";
        break;
    case ReduceOp::Min:
        os() << "min=";
        break;
    case ReduceOp::Max:
        os() << "max=";
        break;
    default:
        ASSERT(false);
    }
    os() << " ";
    recur(op->expr_);
    os() << std::endl;
}

void PrintVisitor::visit(const IntConst &op) {
    if (pretty_) {
        os() << MAGENTA << std::to_string(op->val_) << RESET;
    } else {
        os() << std::to_string(op->val_);
    }
}

void PrintVisitor::visit(const FloatConst &op) {
    if (pretty_) {
        os() << MAGENTA << std::to_string(op->val_) << RESET;
    } else {
        os() << std::to_string(op->val_);
    }
}

void PrintVisitor::visit(const BoolConst &op) {
    if (pretty_) {
        os() << MAGENTA << (op->val_ ? "true" : "false") << RESET;
    } else {
        os() << (op->val_ ? "true" : "false");
    }
}

void PrintVisitor::visit(const Add &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " + ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Sub &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " - ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Mul &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " * ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const RealDiv &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " / ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const FloorDiv &op) {
    os() << "floor(";
    recur(op->lhs_);
    os() << " / ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const CeilDiv &op) {
    os() << "ceil(";
    recur(op->lhs_);
    os() << " / ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const RoundTowards0Div &op) {
    os() << "towards0(";
    recur(op->lhs_);
    os() << " / ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Mod &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " % ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Min &op) {
    os() << "min(";
    recur(op->lhs_);
    os() << ", ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Max &op) {
    os() << "max(";
    recur(op->lhs_);
    os() << ", ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LT &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " < ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LE &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " <= ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const GT &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " > ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const GE &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " >= ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const EQ &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " == ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const NE &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " != ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LAnd &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " && ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LOr &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " || ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const LNot &op) {
    os() << "!";
    recur(op->expr_);
}

void PrintVisitor::visit(const Sqrt &op) {
    os() << "sqrt(";
    recur(op->expr_);
    os() << ")";
}

void PrintVisitor::visit(const Exp &op) {
    os() << "exp(";
    recur(op->expr_);
    os() << ")";
}

void PrintVisitor::visit(const Square &op) {
    os() << "(";
    recur(op->expr_);
    os() << ")^2";
}

void PrintVisitor::visit(const Abs &op) {
    os() << "abs(";
    recur(op->expr_);
    os() << ")";
}

void PrintVisitor::visit(const Floor &op) {
    os() << "floor(";
    recur(op->expr_);
    os() << ")";
}

void PrintVisitor::visit(const Ceil &op) {
    os() << "ceil(";
    recur(op->expr_);
    os() << ")";
}

void PrintVisitor::visit(const IfExpr &op) {
    os() << "(";
    recur(op->cond_);
    os() << " ? ";
    recur(op->thenCase_);
    os() << " : ";
    recur(op->elseCase_);
    os() << ")";
}

void PrintVisitor::visit(const Cast &op) {
    os() << ::ir::toString(op->dtype_) << "(";
    recur(op->expr_);
    os() << ")";
}

void PrintVisitor::visit(const For &op) {
    if (op->noDeps_) {
        makeIndent();
        os() << "// no dependency" << std::endl;
    }
    if (!op->property_.parallel_.empty()) {
        makeIndent();
        os() << "// parallel = " << op->property_.parallel_ << std::endl;
    }
    for (auto &&reduction : op->property_.reductions_) {
        makeIndent();
        os() << "// reduction ";
        switch (reduction.first) {
        case ReduceOp::Add:
            os() << "+: ";
            break;
        case ReduceOp::Mul:
            os() << "*: ";
            break;
        case ReduceOp::Min:
            os() << "min: ";
            break;
        case ReduceOp::Max:
            os() << "max: ";
            break;
        default:
            ASSERT(false);
        }
        recur(reduction.second);
        os() << std::endl;
    }
    if (op->property_.unroll_) {
        makeIndent();
        os() << "// unroll" << std::endl;
    }
    if (op->property_.vectorize_) {
        makeIndent();
        os() << "// vectorize" << std::endl;
    }
    makeIndent();
    if (pretty_) {
        os() << BOLD << "for " << RESET << op->iter_ << " = ";
    } else {
        os() << "for " << op->iter_ << " = ";
    }
    recur(op->begin_);
    os() << " to ";
    recur(op->end_);
    os() << " ";
    beginBlock();
    recur(op->body_);
    endBlock();
}

void PrintVisitor::visit(const If &op) {
    makeIndent();
    if (pretty_) {
        os() << BOLD << "if " << RESET;
    } else {
        os() << "if ";
    }
    recur(op->cond_);
    os() << " ";
    beginBlock();
    recur(op->thenCase_);
    endBlock();
    if (op->elseCase_.isValid()) {
        makeIndent();
        if (pretty_) {
            os() << BOLD << "else " << RESET;
        } else {
            os() << "else ";
        }
        beginBlock();
        recur(op->elseCase_);
        endBlock();
    }
}

void PrintVisitor::visit(const Assert &op) {
    makeIndent();
    os() << "assert ";
    recur(op->cond_);
    os() << " ";
    beginBlock();
    recur(op->body_);
    endBlock();
}

void PrintVisitor::visit(const Intrinsic &op) {
    os() << "intrinsic(\"";
    int i = 0;
    for (char c : op->format_) {
        if (c == '%') {
            recur(op->params_.at(i++));
        } else {
            os() << c;
        }
    }
    os() << "\")";
}

void PrintVisitor::visit(const Eval &op) {
    makeIndent();
    recur(op->expr_);
    os() << std::endl;
}

void PrintVisitor::visit(const MatMul &op) {
    makeIndent();
    os() << "matmul(&";
    recur(op->a_);
    os() << ", &";
    recur(op->b_);
    os() << ", &";
    recur(op->c_);
    os() << ", ";
    recur(op->alpha_);
    os() << ", ";
    recur(op->beta_);
    os() << ", ";
    recur(op->m_);
    os() << ", ";
    recur(op->k_);
    os() << ", ";
    recur(op->n_);
    os() << ", ";
    recur(op->lda_);
    os() << ", ";
    recur(op->ldb_);
    os() << ", ";
    recur(op->ldc_);
    os() << ", ";
    recur(op->stridea_);
    os() << ", ";
    recur(op->strideb_);
    os() << ", ";
    recur(op->stridec_);
    os() << ", ";
    recur(op->batchSize_);
    os() << ", " << op->aIsRowMajor_ << ", " << op->bIsRowMajor_ << ", "
         << op->cIsRowMajor_ << ") <==> ";
    beginBlock();
    recur(op->equivalent_);
    endBlock();
}

std::string toString(const AST &op, bool pretty) {
    PrintVisitor visitor(pretty);
    visitor(op);
    return visitor.toString(
        [](const CodeGenStream &stream) { return stream.os_.str(); });
}

} // namespace ir
