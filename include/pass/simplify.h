#ifndef SIMPLIFY_H
#define SIMPLIFY_H

#include <unordered_map>

#include <mutator.h>

namespace ir {

class SimplifyPass : public Mutator {
    const std::unordered_map<const ExprNode *, uint64_t> &hash_; // expr -> hash
    const std::unordered_map<uint64_t, Expr> &lower_, &upper_; // hash -> bound
    bool isFixPoint_ = true;

  public:
    SimplifyPass(const std::unordered_map<const ExprNode *, uint64_t> &hash,
                 const std::unordered_map<uint64_t, Expr> &lower,
                 const std::unordered_map<uint64_t, Expr> &upper)
        : hash_(hash), lower_(lower), upper_(upper) {}

    bool isFixPoint() const { return isFixPoint_; }

  private:
    uint64_t getHash(const Expr &op);

    template <class T> Expr doSimplify(const T &_op) {
        auto op = Mutator::visit(_op);
        auto h = getHash(op);
        if (lower_.count(h) && upper_.count(h)) {
            auto lower = lower_.at(h);
            auto upper = upper_.at(h);
            auto hl = getHash(lower);
            auto hr = getHash(upper);
            if (hl == hr) {
                if (hl != h) {
                    isFixPoint_ = false;
                }
                return lower;
            }
        }
        return op;
    }

    bool alwaysLT(const Expr &lhs, const Expr &rhs);
    bool alwaysLE(const Expr &lhs, const Expr &rhs);

  protected:
    Expr visit(const Var &op) override { return doSimplify(op); }
    Expr visit(const Add &op) override { return doSimplify(op); }
    Expr visit(const Sub &op) override { return doSimplify(op); }
    Expr visit(const Mul &op) override { return doSimplify(op); }
    Expr visit(const Div &op) override { return doSimplify(op); }
    Expr visit(const LT &op) override;
    Expr visit(const LE &op) override;
    Expr visit(const GT &op) override;
    Expr visit(const GE &op) override;
    Expr visit(const EQ &op) override;
    Expr visit(const NE &op) override;
    Stmt visit(const If &op) override;
};

Stmt simplifyPass(const Stmt &op);

} // namespace ir

#endif // SIMPLIFY_H
