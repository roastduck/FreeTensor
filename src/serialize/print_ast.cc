#include <cctype>

#include <itertools.hpp>

#include <config.h>
#include <serialize/print_ast.h>

#include "../codegen/detail/code_gen.h"

namespace freetensor {

using namespace std::string_literals;
#define MAGENTA "\u001b[35m"s
#define CYAN "\u001b[36m"s
#define BLUE "\u001b[34m"s
#define UNDERLINE "\u001b[4m"s
#define RESET "\u001b[0m"s
#define BOLD "\u001b[1m"s

std::string PrintVisitor::prettyIterName(const std::string &name) {
    auto escaped = escape(name);
    if (pretty_)
        return UNDERLINE + escaped + RESET;
    else
        return escaped;
}

std::string PrintVisitor::prettyVarDefName(const std::string &name) {
    auto escaped = escape(name);
    if (pretty_)
        return BOLD + escaped + RESET;
    else
        return escaped;
}

std::string PrintVisitor::prettyFuncName(const std::string &name) {
    auto escaped = escape(name);
    if (pretty_)
        return BOLD + escaped + RESET;
    else
        return escaped;
}

std::string PrintVisitor::prettyId(const std::string &id) {
    auto escaped = escape(id);
    if (pretty_)
        return CYAN + escaped + RESET;
    else
        return escaped;
}

std::string PrintVisitor::prettyLiteral(const std::string &lit) {
    if (pretty_)
        return MAGENTA + lit + RESET;
    else
        return lit;
}

std::string PrintVisitor::prettyKeyword(const std::string &kw) {
    if (pretty_)
        return BLUE + BOLD + kw + RESET;
    else
        return kw;
}

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

void PrintVisitor::printMetadataAndId(const Stmt &op) {
#ifdef FT_DEBUG_LOG_NODE
    makeIndent();
    os() << "// By " << op->debugCreator_ << std::endl;
#endif
    if (printAllId_) {
        makeIndent();
        os() << "#" << prettyId(::freetensor::toString(op->id()));
    }
    os() << manipMetadataSkipLocation << manipMetadataOneLine << op->metadata()
         << ":" << std::endl;
}

std::string PrintVisitor::escape(const std::string &name) {
    ASSERT(!name.empty());

    bool should_escape = false;
    if (keywords.count(name))
        should_escape = true;
    else if (!isalpha(name[0]) && name[0] != '_')
        should_escape = true;
    else
        for (size_t i = 1, n = name.length(); i < n; i++)
            if (!isalnum(name[i]) && name[i] != '_') {
                should_escape = true;
                break;
            }

    if (should_escape)
        return '`' + name + '`';
    else
        return name;
}

void PrintVisitor::visitStmt(const Stmt &op) {
    if (op->nodeType() != ASTNodeType::Any) {
        printMetadataAndId(op);
    }
    Visitor::visitStmt(op);
}

void PrintVisitor::visit(const Func &op) {
    makeIndent();
    os() << "func " << prettyFuncName(op->name_) << "(";
    for (auto &&[i, param] : iter::enumerate(op->params_)) {
        os() << (i > 0 ? ", " : "") << prettyVarDefName(param.name_);
        if (param.closure_.isValid()) {
            os() << " @!closure /* " << param.closure_.get() << " */";
        }
    }
    os() << ") ";
    if (!op->returns_.empty()) {
        os() << "-> ";
        for (auto &&[i, ret] : iter::enumerate(op->returns_)) {
            auto &&[name, dtype, closure, returnClosure] = ret;
            os() << (i > 0 ? ", " : "") << prettyVarDefName(name) << ": "
                 << ::freetensor::toString(dtype);
            if (closure.isValid()) {
                os() << " @!closure /* " << closure.get() << " */";
            }
        }
        os() << " ";
    }
    beginBlock();
    recur(op->body_);
    endBlock();
}

void PrintVisitor::visit(const StmtSeq &op) {
    if (printAllId_ || op->metadata().isValid()) {
        makeIndent();
        beginBlock();
    }
    if (op->stmts_.empty()) {
        makeIndent();
        os() << "/* empty */" << std::endl;
    } else {
        Visitor::visit(op);
    }
    if (printAllId_ || op->metadata().isValid()) {
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
         << prettyVarDefName(op->name_) << ": ";
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
    os() << prettyIterName(op->name_);
    Visitor::visit(op);
}

void PrintVisitor::visit(const Store &op) {
    makeIndent();
    os() << prettyVarDefName(op->var_) << "[";
    printList(op->indices_);
    os() << "] = ";
    recur(op->expr_);
    os() << std::endl;
}

void PrintVisitor::visit(const Alloc &op) {
    makeIndent();
    os() << "@!alloc(" << prettyVarDefName(op->var_) << ")";
    os() << std::endl;
}

void PrintVisitor::visit(const Free &op) {
    makeIndent();
    os() << "@!free(" << prettyVarDefName(op->var_) << ")";
    os() << std::endl;
}

void PrintVisitor::visit(const Load &op) {
    os() << prettyVarDefName(op->var_) << "[";
    printList(op->indices_);
    os() << "]";
    if (dtypeInLoad_) {
        os() << " : " << ::freetensor::toString(op->loadType_);
    }
}

void PrintVisitor::visit(const ReduceTo &op) {
    if (op->atomic_) {
        makeIndent();
        os() << "@!atomic" << std::endl;
    }
    makeIndent();
    os() << prettyVarDefName(op->var_) << "[";
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
    case ReduceOp::LAnd:
        os() << "&&=";
        break;
    case ReduceOp::LOr:
        os() << "||=";
        break;
    default:
        ASSERT(false);
    }
    os() << " ";
    recur(op->expr_);
    os() << std::endl;
}

void PrintVisitor::visit(const IntConst &op) {
    os() << prettyLiteral(std::to_string(op->val_));
}

void PrintVisitor::visit(const FloatConst &op) {
    os() << prettyLiteral(std::to_string(op->val_));
}

void PrintVisitor::visit(const BoolConst &op) {
    os() << prettyLiteral(op->val_ ? "true" : "false");
}

void PrintVisitor::visit(const Add &op) {
    precedence_enclose(Precedence::ADD, [&] {
        recur(op->lhs_);
        os() << " + ";
        precedence_enclose(Precedence::ADD_RHS, [&] { recur(op->rhs_); });
    });
}

void PrintVisitor::visit(const Sub &op) {
    precedence_enclose(Precedence::ADD, [&] {
        recur(op->lhs_);
        os() << " - ";
        precedence_enclose(Precedence::ADD_RHS, [&] { recur(op->rhs_); });
    });
}

void PrintVisitor::visit(const Mul &op) {
    precedence_enclose(Precedence::MUL, [&] {
        recur(op->lhs_);
        os() << " * ";
        precedence_enclose(Precedence::MUL_RHS, [&] { recur(op->rhs_); });
    });
}

void PrintVisitor::visit(const RealDiv &op) {
    precedence_enclose(Precedence::MUL, [&] {
        recur(op->lhs_);
        os() << " / ";
        precedence_enclose(Precedence::MUL_RHS, [&] { recur(op->rhs_); });
    });
}

void PrintVisitor::visit(const FloorDiv &op) {
    precedence_enclose(
        Precedence::MUL_RHS,
        [&] {
            os() << "@!floor(";
            recur(op->lhs_);
            os() << " / ";
            recur(op->rhs_);
            os() << ")";
        },
        false);
}

void PrintVisitor::visit(const CeilDiv &op) {
    precedence_enclose(
        Precedence::MUL_RHS,
        [&] {
            os() << "@!ceil(";
            recur(op->lhs_);
            os() << " / ";
            recur(op->rhs_);
            os() << ")";
        },
        false);
}

void PrintVisitor::visit(const RoundTowards0Div &op) {
    precedence_enclose(
        Precedence::MUL_RHS,
        [&] {
            os() << "@!towards0(";
            recur(op->lhs_);
            os() << " / ";
            recur(op->rhs_);
            os() << ")";
        },
        false);
}

void PrintVisitor::visit(const Mod &op) {
    precedence_enclose(Precedence::MUL, [&] {
        recur(op->lhs_);
        os() << " % ";
        precedence_enclose(Precedence::MUL_RHS, [&] { recur(op->rhs_); });
    });
}

void PrintVisitor::visit(const Remainder &op) {
    precedence_enclose(Precedence::MUL, [&] {
        recur(op->lhs_);
        os() << " %% ";
        precedence_enclose(Precedence::MUL_RHS, [&] { recur(op->rhs_); });
    });
}

void PrintVisitor::visit(const Min &op) {
    precedence_new([&] {
        os() << "@!min(";
        recur(op->lhs_);
        os() << ", ";
        recur(op->rhs_);
        os() << ")";
    });
}

void PrintVisitor::visit(const Max &op) {
    precedence_new([&] {
        os() << "@!max(";
        recur(op->lhs_);
        os() << ", ";
        recur(op->rhs_);
        os() << ")";
    });
}

void PrintVisitor::visit(const LT &op) {
    precedence_enclose(Precedence::COMP, [&] {
        recur(op->lhs_);
        os() << " < ";
        precedence_enclose(Precedence::COMP_RHS, [&] { recur(op->rhs_); });
    });
}

void PrintVisitor::visit(const LE &op) {
    precedence_enclose(Precedence::COMP, [&] {
        recur(op->lhs_);
        os() << " <= ";
        precedence_enclose(Precedence::COMP_RHS, [&] { recur(op->rhs_); });
    });
}

void PrintVisitor::visit(const GT &op) {
    precedence_enclose(Precedence::COMP, [&] {
        recur(op->lhs_);
        os() << " > ";
        precedence_enclose(Precedence::COMP_RHS, [&] { recur(op->rhs_); });
    });
}

void PrintVisitor::visit(const GE &op) {
    precedence_enclose(Precedence::COMP, [&] {
        recur(op->lhs_);
        os() << " >= ";
        precedence_enclose(Precedence::COMP_RHS, [&] { recur(op->rhs_); });
    });
}

void PrintVisitor::visit(const EQ &op) {
    precedence_enclose(Precedence::COMP, [&] {
        recur(op->lhs_);
        os() << " == ";
        precedence_enclose(Precedence::COMP_RHS, [&] { recur(op->rhs_); });
    });
}

void PrintVisitor::visit(const NE &op) {
    precedence_enclose(Precedence::COMP, [&] {
        recur(op->lhs_);
        os() << " != ";
        precedence_enclose(Precedence::COMP_RHS, [&] { recur(op->rhs_); });
    });
}

void PrintVisitor::visit(const LAnd &op) {
    precedence_enclose(Precedence::LAND, [&] {
        recur(op->lhs_);
        os() << " && ";
        precedence_enclose(Precedence::LAND_RHS, [&] { recur(op->rhs_); });
    });
}

void PrintVisitor::visit(const LOr &op) {
    precedence_enclose(Precedence::LOR, [&] {
        recur(op->lhs_);
        os() << " || ";
        precedence_enclose(Precedence::LOR_RHS, [&] { recur(op->rhs_); });
    });
}

void PrintVisitor::visit(const LNot &op) {
    os() << "!";
    precedence_enclose(Precedence::UNARY_LOGIC, [&] { recur(op->expr_); });
}

void PrintVisitor::visit(const Sqrt &op) {
    precedence_new([&] {
        os() << "@!sqrt(";
        recur(op->expr_);
        os() << ")";
    });
}

void PrintVisitor::visit(const Exp &op) {
    precedence_new([&] {
        os() << "@!exp(";
        recur(op->expr_);
        os() << ")";
    });
}

void PrintVisitor::visit(const Square &op) {
    precedence_new([&] {
        os() << "@!square(";
        recur(op->expr_);
        os() << ")";
    });
}

void PrintVisitor::visit(const Sigmoid &op) {
    precedence_new([&] {
        os() << "@!sigmoid(";
        recur(op->expr_);
        os() << ")";
    });
}

void PrintVisitor::visit(const Tanh &op) {
    precedence_new([&] {
        os() << "@!tanh(";
        recur(op->expr_);
        os() << ")";
    });
}

void PrintVisitor::visit(const Abs &op) {
    precedence_new([&] {
        os() << "@!abs(";
        recur(op->expr_);
        os() << ")";
    });
}

void PrintVisitor::visit(const Floor &op) {
    precedence_new([&] {
        os() << "@!floor(";
        recur(op->expr_);
        os() << ")";
    });
}

void PrintVisitor::visit(const Ceil &op) {
    precedence_new([&] {
        os() << "@!ceil(";
        recur(op->expr_);
        os() << ")";
    });
}

void PrintVisitor::visit(const IfExpr &op) {
    precedence_enclose(Precedence::TRINARY, [&] {
        recur(op->cond_);
        os() << " ? ";
        recur(op->thenCase_);
        os() << " : ";
        recur(op->elseCase_);
    });
}

void PrintVisitor::visit(const Cast &op) {
    precedence_new([&] {
        os() << ::freetensor::toString(op->destType_) << "(";
        recur(op->expr_);
        os() << ")";
    });
}

void PrintVisitor::visit(const For &op) {
    if (!op->property_->noDeps_.empty()) {
        makeIndent();
        os() << "@!no_deps : ";
        for (auto &&[i, var] : iter::enumerate(op->property_->noDeps_)) {
            os() << (i == 0 ? "" : ", ");
            os() << prettyVarDefName(var);
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

        os() << prettyVarDefName(reduction->var_);
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
    os() << prettyKeyword("for ") << prettyIterName(op->iter_)
         << prettyKeyword(" in ");
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
    os() << prettyKeyword("if ");
    recur(op->cond_);
    os() << " ";
    beginBlock();
    recur(op->thenCase_);
    endBlock();
    if (op->elseCase_.isValid()) {
        makeIndent();
        os() << prettyKeyword("else ");
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
    os() << "@!eval(";
    recur(op->expr_);
    os() << ")" << std::endl;
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
    return toString(op, pretty, printAllId, false);
}

std::string toString(const AST &op, bool pretty, bool printAllId,
                     bool dtypeInLoad) {
    PrintVisitor visitor(printAllId, pretty, dtypeInLoad);
    visitor(op);
    return visitor.toString(
        [](const CodeGenStream &stream) { return stream.os_.str(); });
}

int OSTREAM_NO_PRETTY = std::ostream::xalloc();

std::ostream &operator<<(std::ostream &os, const AST &op) {
    if (os.iword(OSTREAM_NO_PRETTY)) {
        os << toString(op, false); // Disable pretty print
    } else {
        os << toString(op); // Follow Config::prettyPrint()
    }
    return os;
}

} // namespace freetensor
