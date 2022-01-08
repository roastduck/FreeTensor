#ifndef GEN_PB_EXPR_H
#define GEN_PB_EXPR_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/hash.h>
#include <math/presburger.h>
#include <visitor.h>

namespace ir {

/**
 * Serialize expressions to an Presburger expression string
 *
 * Returns nullptr for non-Presburger expressions
 */
class GenPBExpr : public Visitor {
  public:
    // hash -> (expr, presburger name)
    typedef std::unordered_map<uint64_t, std::pair<Expr, std::string>> VarMap;

  private:
    std::unordered_map<Expr, std::string> results_;
    std::unordered_set<Expr> visited_;
    std::unordered_map<Expr, int> constants_;
    std::unordered_map<Expr, VarMap> vars_;
    GetHash getHash_;
    Expr parent_ = nullptr;
    std::string varSuffix_;

  public:
    GenPBExpr(const std::string &varSuffix = "") : varSuffix_(varSuffix) {}

    const VarMap &vars(const Expr &op) { return vars_[op]; }

    Ref<std::string> gen(const Expr &op);

  protected:
    void visitExpr(const Expr &op) override;
    void visit(const Var &op) override;
    void visit(const Load &op) override;
    void visit(const IntConst &op) override;
    void visit(const Add &op) override;
    void visit(const Sub &op) override;
    void visit(const Mul &op) override;
    void visit(const LAnd &op) override;
    void visit(const LOr &op) override;
    void visit(const LNot &op) override;
    void visit(const LT &op) override;
    void visit(const LE &op) override;
    void visit(const GT &op) override;
    void visit(const GE &op) override;
    void visit(const EQ &op) override;
    void visit(const NE &op) override;
    void visit(const FloorDiv &op) override;
    void visit(const CeilDiv &op) override;
    void visit(const Mod &op) override;
    void visit(const Min &op) override;
    void visit(const Max &op) override;
};

} // namespace ir

#endif // GEN_PB_EXPR_H
