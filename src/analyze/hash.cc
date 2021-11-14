#include <analyze/hash.h>

namespace ir {

void GetHash::visitExpr(const Expr &op,
                        const std::function<void(const Expr &)> &visitNode) {
    if (!hash_.count(op)) {
        Visitor::visitExpr(op, visitNode);
    }
}

void GetHash::visit(const Var &op) {
    Visitor::visit(op);
    uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
    h = ((h + std::hash<std::string>()(op->name_)) * K2 + B2) % P;
    hash_[op] = h = (h * K3 + B3) % P;
}

void GetHash::visit(const Load &op) {
    Visitor::visit(op);
    uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
    h = ((h + std::hash<std::string>()(op->var_)) * K2 + B2) % P;
    for (auto &&index : op->indices_) {
        h = ((h + hash_.at(index)) * K2 + B2) % P;
    }
    hash_[op] = h = (h * K3 + B3) % P;
}

void GetHash::visit(const IntConst &op) {
    Visitor::visit(op);
    uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
    h = ((h + std::hash<int>()(op->val_)) * K2 + B2) % P;
    hash_[op] = h = (h * K3 + B3) % P;
}

void GetHash::visit(const FloatConst &op) {
    Visitor::visit(op);
    uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
    h = ((h + std::hash<double>()(op->val_)) * K2 + B2) % P;
    hash_[op] = h = (h * K3 + B3) % P;
}

void GetHash::visit(const BoolConst &op) {
    Visitor::visit(op);
    uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
    h = ((h + std::hash<bool>()(op->val_)) * K2 + B2) % P;
    hash_[op] = h = (h * K3 + B3) % P;
}

void GetHash::visit(const Add &op) { binOpPermutable(op); }
void GetHash::visit(const Sub &op) { binOpNonPermutable(op); }
void GetHash::visit(const Mul &op) { binOpPermutable(op); }
void GetHash::visit(const RealDiv &op) { binOpNonPermutable(op); }
void GetHash::visit(const FloorDiv &op) { binOpNonPermutable(op); }
void GetHash::visit(const CeilDiv &op) { binOpNonPermutable(op); }
void GetHash::visit(const RoundTowards0Div &op) { binOpNonPermutable(op); }
void GetHash::visit(const Mod &op) { binOpNonPermutable(op); }
void GetHash::visit(const Min &op) { binOpPermutable(op); }
void GetHash::visit(const Max &op) { binOpPermutable(op); }
void GetHash::visit(const LT &op) { binOpNonPermutable(op); }
void GetHash::visit(const LE &op) { binOpNonPermutable(op); }
void GetHash::visit(const GT &op) { binOpNonPermutable(op); }
void GetHash::visit(const GE &op) { binOpNonPermutable(op); }
void GetHash::visit(const EQ &op) { binOpPermutable(op); }
void GetHash::visit(const NE &op) { binOpPermutable(op); }
void GetHash::visit(const LAnd &op) { binOpPermutable(op); }
void GetHash::visit(const LOr &op) { binOpPermutable(op); }
void GetHash::visit(const LNot &op) { unaryOp(op); }
void GetHash::visit(const Sqrt &op) { unaryOp(op); }
void GetHash::visit(const Exp &op) { unaryOp(op); }
void GetHash::visit(const Square &op) { unaryOp(op); }
void GetHash::visit(const Sigmoid &op) { unaryOp(op); }
void GetHash::visit(const Tanh &op) { unaryOp(op); }
void GetHash::visit(const Abs &op) { unaryOp(op); }
void GetHash::visit(const Floor &op) { unaryOp(op); }
void GetHash::visit(const Ceil &op) { unaryOp(op); }

void GetHash::visit(const IfExpr &op) {
    Visitor::visit(op);
    uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
    h = ((h + hash_.at(op->cond_)) * K2 + B2) % P;
    h = ((h + hash_.at(op->thenCase_)) * K2 + B2) % P;
    h = ((h + hash_.at(op->elseCase_)) * K2 + B2) % P;
    hash_[op] = h = (h * K3 + B3) % P;
}

void GetHash::visit(const Cast &op) {
    Visitor::visit(op);
    uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
    h = ((h + hash_.at(op->expr_)) * K2 + B2) % P;
    h = ((h + std::hash<int>()(int(op->dtype_))) * K2 + B2) % P;
    hash_[op] = h = (h * K3 + B3) % P;
}

void GetHash::visit(const Intrinsic &op) {
    Visitor::visit(op);
    uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
    h = ((h + std::hash<std::string>()(op->format_)) * K2 + B2) % P;
    for (auto &&item : op->params_) {
        h = ((h + hash_.at(item)) * K2 + B2) % P;
    }
    hash_[op] = h = (h * K3 + B3) % P;
}

} // namespace ir

