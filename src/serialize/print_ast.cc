#include <cctype>

#include <itertools.hpp>

#include <config.h>
#include <serialize/print_ast.h>

#include "../codegen/detail/code_gen.h"

namespace freetensor {

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
#ifdef FT_DEBUG_LOG_NODE
    makeIndent();
    os() << "// By " << op->debugCreator_ << std::endl;
#endif
    if (printAllId_ || op->hasNamedId()) {
        if (pretty_) {
            os() << CYAN << printName(::freetensor::toString(op->id())) << ":"
                 << RESET << std::endl;
        } else {
            os() << printName(::freetensor::toString(op->id())) << ":"
                 << std::endl;
        }
    }
}

std::string PrintVisitor::printName(const std::string &name) {
    ASSERT(!name.empty());
    if (keywords.count(name)) {
        goto escape;
    }
    if (!isalpha(name[0]) && name[0] != '_') {
        goto escape;
    }
    for (size_t i = 1, n = name.length(); i < n; i++) {
        if (!isalnum(name[i]) && name[i] != '_') {
            goto escape;
        }
    }
    return name;
escape:
    return '`' + name + '`';
}

void PrintVisitor::visitStmt(const Stmt &op) {
    if (op->nodeType() != ASTNodeType::Any) {
        printId(op);
    }
    Visitor::visitStmt(op);
}

void PrintVisitor::visit(const Func &op) {
    makeIndent();
    os() << "func " << printName(op->name_) << "(";
    for (auto &&[i, param] : iter::enumerate(op->params_)) {
        os() << (i > 0 ? ", " : "") << printName(param);
        if (op->closure_.count(param)) {
            os() << " @!closure /* " << op->closure_.at(param).get() << " */";
        }
    }
    os() << ") ";
    if (!op->returns_.empty()) {
        os() << "-> ";
        for (auto &&[i, ret] : iter::enumerate(op->returns_)) {
            auto &&[name, dtype] = ret;
            os() << (i > 0 ? ", " : "") << printName(name) << ": "
                 << ::freetensor::toString(dtype);
            if (op->closure_.count(name)) {
                os() << " @!closure /* " << op->closure_.at(name).get()
                     << " */";
            }
        }
        os() << " ";
    }
    beginBlock();
    recur(op->body_);
    endBlock();
}

void PrintVisitor::visit(const StmtSeq &op) {
    if (printAllId_ || op->hasNamedId()) {
        makeIndent();
        beginBlock();
    }
    if (op->stmts_.empty()) {
        makeIndent();
        os() << "/* empty */" << std::endl;
    } else {
        Visitor::visit(op);
    }
    if (printAllId_ || op->hasNamedId()) {
        endBlock();
    }
}

void PrintVisitor::visit(const Any &op) {
    makeIndent();
    os() << "<Any>" << std::endl;
}

void PrintVisitor::visit(const AnyExpr &op) { os() << "<Any>"; }

void PrintVisitor::visit(const VarDef &op) {
    makeIndent();
    os() << "@" << ::freetensor::toString(op->buffer_->atype()) << " @"
         << ::freetensor::toString(op->buffer_->mtype()) << " "
         << printName(op->name_) << ": ";
    auto &&tensor = op->buffer_->tensor();
    os() << ::freetensor::toString(tensor->dtype()) << "[";
    printList(tensor->shape());
    os() << "] ";
    if (op->ioTensor_.isValid()) {
        os() << "@!io_tensor = ";
        os() << ::freetensor::toString(op->ioTensor_->dtype()) << "[";
        printList(op->ioTensor_->shape());
        os() << "] ";
        os() << " ";
    }
    if (op->pinned_) {
        os() << "@!pinned ";
    }
    beginBlock();
    recur(op->body_);
    endBlock();
}

void PrintVisitor::visit(const Var &op) {
    os() << printName(op->name_);
    Visitor::visit(op);
}

void PrintVisitor::visit(const Store &op) {
    makeIndent();
    os() << printName(op->var_) << "[";
    printList(op->indices_);
    os() << "] = ";
    recur(op->expr_);
    os() << std::endl;
}

void PrintVisitor::visit(const Load &op) {
    os() << printName(op->var_) << "[";
    printList(op->indices_);
    os() << "]";
}

void PrintVisitor::visit(const ReduceTo &op) {
    if (op->atomic_) {
        makeIndent();
        os() << "@!atomic" << std::endl;
    }
    makeIndent();
    os() << printName(op->var_) << "[";
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
        os() << "@!min=";
        break;
    case ReduceOp::Max:
        os() << "@!max=";
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
    os() << "@!floor(";
    recur(op->lhs_);
    os() << " / ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const CeilDiv &op) {
    os() << "@!ceil(";
    recur(op->lhs_);
    os() << " / ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const RoundTowards0Div &op) {
    os() << "@!towards0(";
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

void PrintVisitor::visit(const Remainder &op) {
    os() << "(";
    recur(op->lhs_);
    os() << " %% ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Min &op) {
    os() << "@!min(";
    recur(op->lhs_);
    os() << ", ";
    recur(op->rhs_);
    os() << ")";
}

void PrintVisitor::visit(const Max &op) {
    os() << "@!max(";
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
    os() << "@!sqrt(";
    recur(op->expr_);
    os() << ")";
}

void PrintVisitor::visit(const Exp &op) {
    os() << "@!exp(";
    recur(op->expr_);
    os() << ")";
}

void PrintVisitor::visit(const Square &op) {
    os() << "@!square(";
    recur(op->expr_);
    os() << ")";
}

void PrintVisitor::visit(const Sigmoid &op) {
    os() << "@!sigmoid(";
    recur(op->expr_);
    os() << ")";
}

void PrintVisitor::visit(const Tanh &op) {
    os() << "@!tanh(";
    recur(op->expr_);
    os() << ")";
}

void PrintVisitor::visit(const Abs &op) {
    os() << "@!abs(";
    recur(op->expr_);
    os() << ")";
}

void PrintVisitor::visit(const Floor &op) {
    os() << "@!floor(";
    recur(op->expr_);
    os() << ")";
}

void PrintVisitor::visit(const Ceil &op) {
    os() << "@!ceil(";
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
    os() << ::freetensor::toString(op->dtype_) << "(";
    recur(op->expr_);
    os() << ")";
}

void PrintVisitor::visit(const For &op) {
    if (!op->property_->noDeps_.empty()) {
        makeIndent();
        os() << "@!no_deps : ";
        for (auto &&[i, var] : iter::enumerate(op->property_->noDeps_)) {
            os() << (i == 0 ? "" : ", ");
            os() << printName(var);
        }
        os() << std::endl;
    }
    if (auto str = ::freetensor::toString(op->property_->parallel_);
        !str.empty()) {
        makeIndent();
        os() << "@!parallel : @" << str << std::endl;
    }
    for (auto &&reduction : op->property_->reductions_) {
        makeIndent();
        os() << "@!reduction ";
        switch (reduction->op_) {
        case ReduceOp::Add:
            os() << "+= ";
            break;
        case ReduceOp::Mul:
            os() << "*= ";
            break;
        case ReduceOp::Min:
            os() << "@!min= ";
            break;
        case ReduceOp::Max:
            os() << "@!max= ";
            break;
        default:
            ASSERT(false);
        }
        os() << ": ";

        os() << printName(reduction->var_);
        for (auto &&[b, e] : iter::zip(reduction->begins_, reduction->ends_)) {
            os() << "[";
            (*this)(b);
            os() << ":";
            (*this)(e);
            os() << "]";
        }
        os() << std::endl;
    }
    if (op->property_->unroll_) {
        makeIndent();
        os() << "@!unroll" << std::endl;
    }
    if (op->property_->vectorize_) {
        makeIndent();
        os() << "@!vectorize" << std::endl;
    }
    if (op->property_->preferLibs_) {
        makeIndent();
        os() << "@!prefer_libs" << std::endl;
    }
    makeIndent();
    if (pretty_) {
        os() << BOLD << "for " << RESET << op->iter_ << " in ";
    } else {
        os() << "for " << op->iter_ << " in ";
    }
    recur(op->begin_);
    os() << " : ";
    recur(op->end_);
    os() << " : ";
    recur(op->step_);
    os() << " : ";
    recur(op->len_);
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

void PrintVisitor::visit(const Assume &op) {
    makeIndent();
    os() << "assume ";
    recur(op->cond_);
    os() << " ";
    beginBlock();
    recur(op->body_);
    endBlock();
}

void PrintVisitor::visit(const Intrinsic &op) {
    os() << "@!intrinsic(\"" << op->format_ << "\" -> "
         << ::freetensor::toString(op->retType_);
    for (auto &&param : op->params_) {
        os() << ", ";
        recur(param);
    }
    if (op->hasSideEffect_) {
        os() << ", @!side_effect";
    }
    os() << ")";
}

void PrintVisitor::visit(const Eval &op) {
    makeIndent();
    recur(op->expr_);
    os() << std::endl;
}

void PrintVisitor::visit(const MatMul &op) {
    makeIndent();
    os() << "@!matmul(&";
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

std::string toString(const AST &op) {
    return toString(op, Config::prettyPrint());
}

std::string toString(const AST &op, bool pretty) {
    return toString(op, pretty, Config::printAllId());
}

std::string toString(const AST &op, bool pretty, bool printAllId) {
    PrintVisitor visitor(printAllId, pretty);
    visitor(op);
    return visitor.toString(
        [](const CodeGenStream &stream) { return stream.os_.str(); });
}

} // namespace freetensor
