#ifndef FLOAT_SIMPLIFY_H
#define FLOAT_SIMPLIFY_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/hash.h>
#include <analyze/type_infer.h>
#include <func.h>
#include <mutator.h>

namespace ir {

class FloatSimplify : public Mutator {
    std::unordered_map<Expr, double> constants_;
    std::unordered_set<Expr> nonNeg_, nonPosi_;
    std::unordered_map<std::string, Ref<Buffer>> buffers_;
    GetHash getHash_;
    TypeInfer typeInfer_;
    bool isFixPoint_ = true;

  public:
    FloatSimplify() : typeInfer_(&buffers_) {}

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
    Stmt visit(const VarDef &op) override;
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

inline Func floatSimplify(const Func &func) {
    return makeFunc(func->name_, func->params_, func->buffers_,
                    floatSimplify(func->body_), func->src_);
}

} // namespace ir

#endif // FLOAT_SIMPLIFY_H
