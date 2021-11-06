#ifndef HASH_H
#define HASH_H

#include <unordered_map>

#include <visitor.h>

namespace ir {

/**
 * Get hash of any (sub)expression
 */
class GetHash : public Visitor {
    std::unordered_map<Expr, uint64_t> hash_;

    static constexpr uint64_t P = 2147483647; // % P
    static constexpr uint64_t K1 = 179424673, B1 = 275604541;
    // (node type * K1 + B1) % P
    static constexpr uint64_t K2 = 373587883, B2 = 472882027;
    // ((current hash + non-permutable factor) * K2 + B2) % P
    // or
    // (current hash + permutable factor) % P
    static constexpr uint64_t K3 = 573259391, B3 = 674506081;
    // (finally * K3 + B3) % P

  private:
    template <class T> void binOpPermutable(const T &op) {
        Visitor::visit(op);
        uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
        // Permutable
        h += hash_.at(op->lhs_) + hash_.at(op->rhs_);
        hash_[op] = h = (h * K3 + B3) % P;
    }

    template <class T> void binOpNonPermutable(const T &op) {
        Visitor::visit(op);
        uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
        // Non-permutable
        h = ((h + hash_.at(op->lhs_)) * K2 + B2) % P;
        h = ((h + hash_.at(op->rhs_)) * K2 + B2) % P;
        hash_[op] = h = (h * K3 + B3) % P;
    }

    template <class T> void unaryOp(const T &op) {
        Visitor::visit(op);
        uint64_t h = ((uint64_t)op->nodeType() * K1 + B1) % P;
        h = ((h + hash_.at(op->expr_)) * K2 + B2) % P;
        hash_[op] = h = (h * K3 + B3) % P;
    }

  protected:
    void visitExpr(const Expr &op,
                   const std::function<void(const Expr &)> &visitNode) override;

    void visit(const Var &op) override;
    void visit(const Load &op) override;
    void visit(const IntConst &op) override;
    void visit(const FloatConst &op) override;
    void visit(const BoolConst &op) override;
    void visit(const Add &op) override;
    void visit(const Sub &op) override;
    void visit(const Mul &op) override;
    void visit(const RealDiv &op) override;
    void visit(const FloorDiv &op) override;
    void visit(const CeilDiv &op) override;
    void visit(const RoundTowards0Div &op) override;
    void visit(const Mod &op) override;
    void visit(const Min &op) override;
    void visit(const Max &op) override;
    void visit(const LT &op) override;
    void visit(const LE &op) override;
    void visit(const GT &op) override;
    void visit(const GE &op) override;
    void visit(const EQ &op) override;
    void visit(const NE &op) override;
    void visit(const LAnd &op) override;
    void visit(const LOr &op) override;
    void visit(const LNot &op) override;
    void visit(const Sqrt &op) override;
    void visit(const Exp &op) override;
    void visit(const Square &op) override;
    void visit(const Sigmoid &op) override;
    void visit(const Abs &op) override;
    void visit(const Floor &op) override;
    void visit(const Ceil &op) override;
    void visit(const IfExpr &op) override;
    void visit(const Cast &op) override;
    void visit(const Intrinsic &op) override;

  public:
    const std::unordered_map<Expr, uint64_t> &hash() const { return hash_; }
};

/**
 * Get hashes of all expressions in an AST
 *
 * Statements are NOT hashed
 */
inline std::unordered_map<Expr, uint64_t> getHashMap(const AST &op) {
    GetHash visitor;
    visitor(op);
    return visitor.hash();
}

inline uint64_t getHash(const Expr &op) { return getHashMap(op).at(op); }

} // namespace ir

#endif // HASH_H
