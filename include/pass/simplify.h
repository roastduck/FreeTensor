#ifndef SIMPLIFY_H
#define SIMPLIFY_H

#include <unordered_map>

#include <mutator.h>

namespace ir {

class SimplifyPass : public Mutator {
    const std::unordered_map<const ExprNode *, uint64_t> &hash_;
    const std::unordered_map<const ExprNode *, std::vector<Expr>> &lower_,
        &upper_;
    bool isFixPoint_ = true;

  public:
    SimplifyPass(
        const std::unordered_map<const ExprNode *, uint64_t> &hash,
        const std::unordered_map<const ExprNode *, std::vector<Expr>> &lower,
        const std::unordered_map<const ExprNode *, std::vector<Expr>> &upper)
        : hash_(hash), lower_(lower), upper_(upper) {}

    bool isFixPoint() const { return isFixPoint_; }

  private:
    uint64_t getHash(const Expr &op);

    template <class T> Expr doSimplify(const T &_op) {
        auto op = Mutator::visit(_op);
        // lower_ / upper_ for _op and op shall be the same, but those for op
        // are not updated, so using _op
        if (lower_.count(_op.get()) && upper_.count(_op.get())) {
            for (auto &&lower : lower_.at(_op.get())) {
                for (auto &&upper : upper_.at(_op.get())) {
                    auto hl = getHash(lower);
                    auto hr = getHash(upper);
                    if (hl == hr) {
                        if (hl != getHash(op)) {
                            isFixPoint_ = false;
                        }
                        return lower;
                        // FIXME: We need to choose the simplest one. Other wise
                        // we are always picking the original expression
                    }
                }
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
