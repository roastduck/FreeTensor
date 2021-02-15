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

  protected:
    virtual void visit(const Var &op) override;
    virtual void visit(const Load &op) override;
    virtual void visit(const IntConst &op) override;
    virtual void visit(const FloatConst &op) override;
    virtual void visit(const Add &op) override;
    virtual void visit(const Sub &op) override;
    virtual void visit(const Mul &op) override;
    virtual void visit(const Div &op) override;
    virtual void visit(const Mod &op) override;
    virtual void visit(const Min &op) override;
    virtual void visit(const Max &op) override;
    virtual void visit(const LT &op) override;
    virtual void visit(const LE &op) override;
    virtual void visit(const GT &op) override;
    virtual void visit(const GE &op) override;
    virtual void visit(const EQ &op) override;
    virtual void visit(const NE &op) override;
    virtual void visit(const LAnd &op) override;
    virtual void visit(const LOr &op) override;
    virtual void visit(const LNot &op) override;

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
