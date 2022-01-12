#ifndef FLOAT_SIMPLIFY_H
#define FLOAT_SIMPLIFY_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/hash.h>
#include <analyze/symbol_table.h>
#include <analyze/type_infer.h>
#include <func.h>
#include <mutator.h>

namespace ir {

class FloatSimplify : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    std::unordered_map<Expr, double> constants_;
    std::unordered_set<Expr> nonNeg_, nonPosi_;
    GetHash getHash_;
    TypeInfer typeInfer_;
    bool isFixPoint_ = true;

  public:
    FloatSimplify() : typeInfer_(*this) {}

    bool isFixPoint() const { return isFixPoint_; }

    void setNonNeg(const Expr &op) { nonNeg_.insert(op); }
    void setNonPosi(const Expr &op) { nonPosi_.insert(op); }
    bool nonNeg(const Expr &op) const {
        return nonNeg_.count(op) ||
               (constants_.count(op) && constants_.at(op) >= 0);
    }
    bool nonPosi(const Expr &op) const {
        return nonPosi_.count(op) ||
               (constants_.count(op) && constants_.at(op) <= 0);
    }

  private:
    uint64_t getHash(const Expr &op);
    DataType dtype(const Expr &op);

    Expr normalizeRealMulDiv(const Expr &op);

  protected:
    using BaseClass::visit;
    Expr visit(const IntConst &op) override;
    Expr visit(const FloatConst &op) override;
    Expr visit(const Add &op) override;
    Expr visit(const Sub &op) override;
    Expr visit(const Mul &op) override;
    Expr visit(const RealDiv &op) override;
    Expr visit(const Min &op) override;
    Expr visit(const Max &op) override;
    Expr visit(const Sqrt &op) override;
    Expr visit(const Exp &op) override;
    Expr visit(const Square &op) override;
    Expr visit(const Abs &op) override;
};

/**
 * Simplify floating-point expressions in an AST
 */
Stmt floatSimplify(const Stmt &op);

DEFINE_PASS_FOR_FUNC(floatSimplify)

} // namespace ir

#endif // FLOAT_SIMPLIFY_H
