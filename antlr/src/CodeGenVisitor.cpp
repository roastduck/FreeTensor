#include "CodeGenVisitor.h"

std::string CodeGenVisitor::genCode(
        const std::shared_ptr<ASTNode> &op,
        const std::unordered_map<std::string, int> &varMap,
        const std::unordered_map<std::string, ExprType> &typeInfo) {
    varMap_ = &varMap, typeInfo_ = &typeInfo;
    (*this)(op);
    return os.str();
}

void CodeGenVisitor::visit(const ProgramNode *op) {
    os << "\n"
          ".section .text\n"
          ".global main\n";
    Visitor::visit(op);
}

void CodeGenVisitor::visit(const FunctionNode *op) {
    if (!op->body_) {
        return;
    }

    curFunc_ = op->name_;
    retTarget_ = jumpCnt_++;
    curFuncNVar_ = 0;
    for (auto &&item : *varMap_) {
        if (item.first.substr(0, curFunc_.length() + 1) == curFunc_ + "/") {
            curFuncNVar_++;
        }
    }

    os << op->name_ << ":\n";
    os << "sd fp, -8(sp)\n"
          "mv fp, sp\n";
    for (size_t i = 0, n = op->args_.size(); i < n; i++) {
        // copy args as new vars
        auto offset = varMap_->at(curFunc_ + "/" + op->args_[i].second);
        os << "ld t0, " << (8 * i) << "(fp)\n"
              "sd t0, " << (-16 - 8 * offset) << "(fp)  # Store to " << op->args_[i].second << "\n";
    }
    Visitor::visit(op);
    os << "mv a0, x0\n";  // step5 requires the default return value to be 0
    os << retTarget_ << ":\n";
    os << "mv sp, fp\n"
          "ld fp, -8(sp)\n"
          "ret\n";
}

void CodeGenVisitor::visit(const VarNode *op) {
    Visitor::visit(op);
    auto fullname = getFullname(op->name_);
    if (varMap_->count(fullname)) {
        auto offset = varMap_->at(fullname);
        os << "ld a0, " << (-16 - 8 * offset) << "(fp)  # Load from " << op->name_ << "\n" << push;
    } else {
        ASSERT(op->name_ == fullname);  // global va
        os << "ld a0, " << op->name_ << "\n" << push;
    }
}

void CodeGenVisitor::visit(const AssignNode *op) {
    auto fullname = getFullname(op->var_);
    ASSERT(op->expr_->type_ == typeInfo_->at(fullname));
    stmtPrelude();
    (*this)(op->expr_);
    if (varMap_->count(fullname)) {
        auto offset = varMap_->at(fullname);
        os << "sd a0, " << (-16 - 8 * offset) << "(fp)  # Store to " << op->var_ << "\n";
    } else {
        ASSERT(op->var_ == fullname);  // global va
        os << "sd a0, " << op->var_ << ", t0\n" << push;  // t0 is a scratch register
    }
}

void CodeGenVisitor::visit(const InvokeNode *op) {
    stmtPrelude();
    Visitor::visit(op);
}

void CodeGenVisitor::visit(const IfThenElseNode *op) {
    stmtPrelude();
    (*this)(op->cond_);
    if (op->elseCase_ == nullptr) {
        auto elseTarget = jumpCnt_++;
        os << "beqz a0, " << elseTarget << "f\n";
        (*this)(op->thenCase_);
        os << elseTarget << ":\n";
    } else {
        auto elseTarget = jumpCnt_++;
        auto endTarget = jumpCnt_++;
        os << "beqz a0, " << elseTarget << "f\n";
        (*this)(op->thenCase_);
        os << "j " << endTarget << "f\n";
        os << elseTarget << ":\n";
        (*this)(op->elseCase_);
        os << endTarget << ":\n";
    }
}

void CodeGenVisitor::visit(const WhileNode *op) {
    stmtPrelude();
    auto beginTarget = continueTarget_ = jumpCnt_++;
    auto endTarget = breakTarget_ = jumpCnt_++;
    os << beginTarget << ":\n";
    (*this)(op->cond_);
    os << "beqz a0, " << endTarget << "f\n";
    (*this)(op->body_);
    os << "j " << beginTarget << "b\n";
    os << endTarget << ":\n";
}

void CodeGenVisitor::visit(const DoWhileNode *op) {
    stmtPrelude();
    auto beginTarget = continueTarget_ = jumpCnt_++;
    auto initTarget = jumpCnt_++;
    auto endTarget = breakTarget_ = jumpCnt_++;
    os << "j " << initTarget << "f\n";
    os << beginTarget << ":\n";
    (*this)(op->cond_);
    os << "beqz a0, " << endTarget << "f\n";
    os << initTarget << ":\n";
    (*this)(op->body_);
    os << "j " << beginTarget << "b\n";
    os << endTarget << ":\n";
}

void CodeGenVisitor::visit(const ForNode *op) {
    stmtPrelude();
    auto beginTarget = continueTarget_ = jumpCnt_++;
    auto noIncrTarget = jumpCnt_++;
    auto endTarget = breakTarget_ = jumpCnt_++;
    (*this)(op->init_);
    os << "j " << noIncrTarget << "f\n";
    os << beginTarget << ":\n";
    (*this)(op->incr_);
    os << noIncrTarget << ":\n";
    (*this)(op->cond_);
    os << "beqz a0, " << endTarget << "f\n";
    (*this)(op->body_);
    os << "j " << beginTarget << "b\n";
    os << endTarget << ":\n";
}

void CodeGenVisitor::visit(const ReturnNode *op) {
    ASSERT(op->expr_->type_ == typeInfo_->at(curFunc_));
    stmtPrelude();
    Visitor::visit(op);
    os << "j " << retTarget_ << "f\n";
}

void CodeGenVisitor::visit(const BreakNode *op) {
    Visitor::visit(op);
    os << "j " << breakTarget_ << "f\n";
}

void CodeGenVisitor::visit(const ContinueNode *op) {
    Visitor::visit(op);
    os << "j " << continueTarget_ << "b\n";
}

void CodeGenVisitor::visit(const IntegerNode *op) {
    Visitor::visit(op);
    os << "ori a0, x0, " << op->literal_ << "\n" << push;
}

void CodeGenVisitor::visit(const CallNode *op) {
    os << "sd ra, -8(sp)\n"
          "sd t0, -16(sp)\n"
          "sd t1, -24(sp)\n"
          "addi sp, sp, -24\n";
    for (size_t i = op->args_.size() - 1; ~i; i--) {
        (*this)(op->args_[i]);  // results are saved to stack
        // TODO: check the types here
    }
    os << "call "  << op->callee_ << "\n";
    os << "addi sp, sp, " << (24 + 8 * op->args_.size()) << "\n"
          "ld ra, -8(sp)\n"
          "ld t0, -16(sp)\n"
          "ld t1, -24(sp)\n";
    os << push;
}

void CodeGenVisitor::visit(const CastNode *op) {
    Visitor::visit(op);
    if (op->expr_->type_ == op->type_) {
        return;
    }
    if (op->expr_->type_ == ExprType::Int && op->type_ == ExprType::Bool) {
        os << "snez a0, a0\n" << puttop;
    } else if (op->expr_->type_ == ExprType::Bool && op->type_ == ExprType::Int) {
        // nothing
    } else {
        throw std::runtime_error("Invalid type conversion");
    }
}

void CodeGenVisitor::visit(const AddNode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "add a0, t0, t1\n" << push;
}

void CodeGenVisitor::visit(const SubNode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "sub a0, t0, t1\n" << push;
}

void CodeGenVisitor::visit(const MulNode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "mul a0, t0, t1\n" << push;
}

void CodeGenVisitor::visit(const DivNode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "div a0, t0, t1\n" << push;
}

void CodeGenVisitor::visit(const ModNode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "rem a0, t0, t1\n" << push;
}

void CodeGenVisitor::visit(const BAndNode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "and a0, t0, t1\n" << push;
}

void CodeGenVisitor::visit(const BOrNode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "or a0, t0, t1\n" << push;
}

void CodeGenVisitor::visit(const BXorNode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "xor a0, t0, t1\n" << push;
}

void CodeGenVisitor::visit(const SLLNode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "sll a0, t0, t1\n" << push;
}

void CodeGenVisitor::visit(const SRANode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "sra a0, t0, t1\n" << push;
}

void CodeGenVisitor::visit(const LNotNode *op) {
    ASSERT(op->expr_->type_ == ExprType::Bool);
    Visitor::visit(op);
    os << "xori a0, a0, 1\n" << puttop;
}

void CodeGenVisitor::visit(const LTNode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "slt a0, t0, t1\n" << push;
}

void CodeGenVisitor::visit(const LENode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "sgt a0, t0, t1\n"
                  "xori a0, a0, 1\n" << push;
}

void CodeGenVisitor::visit(const GTNode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "sgt a0, t0, t1\n" << push;
}

void CodeGenVisitor::visit(const GENode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "slt a0, t0, t1\n"
                  "xori a0, a0, 1\n" << push;
}

void CodeGenVisitor::visit(const EQNode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "sub t0, t0, t1\n"
                  "seqz a0, t0\n" << push;
}

void CodeGenVisitor::visit(const NENode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Int && op->rhs_->type_ == ExprType::Int);
    Visitor::visit(op);
    os << pop2 << "sub t0, t0, t1\n"
                  "snez a0, t0\n" << push;
}

void CodeGenVisitor::visit(const LAndNode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Bool && op->rhs_->type_ == ExprType::Bool);
    Visitor::visit(op);
    os << pop2 << "and a0, t0, t1\n" << push;
}

void CodeGenVisitor::visit(const LOrNode *op) {
    ASSERT(op->lhs_->type_ == ExprType::Bool && op->rhs_->type_ == ExprType::Bool);
    Visitor::visit(op);
    os << pop2 << "or a0, t0, t1\n" << push;
}

void CodeGenVisitor::visit(const SelectNode *op) {
    (*this)(op->cond_);
    auto elseTarget = jumpCnt_++;
    auto endTarget = jumpCnt_++;
    os << "beqz a0, " << elseTarget << "f\n";
    (*this)(op->thenCase_);
    os << "j " << endTarget << "f\n";
    os << elseTarget << ":\n";
    (*this)(op->elseCase_);
    os << endTarget << ":\n";
}

void CodeGenVisitor::stmtPrelude() {
    os << "addi sp, fp, " << (-8 - 8 * curFuncNVar_) << "\n";
}

std::string CodeGenVisitor::getFullname(const std::string &name) const {
    for (auto i = curPath_.length() - 1; ~i; i--) {
        if (curPath_[i] == '/') {
            auto fullname = curPath_.substr(0, i + 1) + name;
            if (typeInfo_->count(fullname)) {
                return fullname;
            }
        }
    }
    if (typeInfo_->count(name)) {
        return name;
    }
    throw std::runtime_error("name " + name + " not found");
}

