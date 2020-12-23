#include <arith/hash.h>

namespace ir {

void GetHash::visit(const Var &op) {
    Visitor::visit(op);
    uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
    h = ((h + std::hash<std::string>()(op->name_)) * K2 + B2) % P;
    hash_[op.get()] = h = (h * K3 + B3) % P;
    subexpr_[h] = op;
}

void GetHash::visit(const Load &op) {
    Visitor::visit(op);
    uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
    h = ((h + std::hash<std::string>()(op->var_)) * K2 + B2) % P;
    for (auto &&index : op->indices_) {
        h = ((h + hash_.at(index.get())) * K2 + B2) % P;
    }
    hash_[op.get()] = h = (h * K3 + B3) % P;
    subexpr_[h] = op;
}

void GetHash::visit(const IntConst &op) {
    Visitor::visit(op);
    uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
    h = ((h + std::hash<int>()(op->val_)) * K2 + B2) % P;
    hash_[op.get()] = h = (h * K3 + B3) % P;
    subexpr_[h] = op;
}

void GetHash::visit(const FloatConst &op) {
    Visitor::visit(op);
    uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
    h = ((h + std::hash<double>()(op->val_)) * K2 + B2) % P;
    hash_[op.get()] = h = (h * K3 + B3) % P;
    subexpr_[h] = op;
}

void GetHash::visit(const Add &op) { binOpPermutable(op); }
void GetHash::visit(const Sub &op) { binOpNonPermutable(op); }
void GetHash::visit(const Mul &op) { binOpPermutable(op); }
void GetHash::visit(const Div &op) { binOpNonPermutable(op); }
void GetHash::visit(const Mod &op) { binOpNonPermutable(op); }
void GetHash::visit(const LT &op) { binOpNonPermutable(op); }
void GetHash::visit(const LE &op) { binOpNonPermutable(op); }
void GetHash::visit(const GT &op) { binOpNonPermutable(op); }
void GetHash::visit(const GE &op) { binOpNonPermutable(op); }
void GetHash::visit(const EQ &op) { binOpPermutable(op); }
void GetHash::visit(const NE &op) { binOpPermutable(op); }

} // namespace ir

