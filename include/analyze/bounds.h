#ifndef BOUNDS_H
#define BOUNDS_H

#include <unordered_map>

#include <analyze/linear.h>
#include <visitor.h>

namespace ir {

struct Bound {
    Expr expr_;
    LinearExpr lin_;

    Bound(const Expr &expr);
    Bound(const LinearExpr &lin);
};

/**
 * Try to get the upper bound and lower bound of each (sub)expression
 *
 * This pass is not accurate. Simplifying passes using thiss analysis may need
 * to run for multiple rounds
 */
class AnalyzeBounds : public Visitor {
  public:
    typedef std::unordered_map<const ExprNode *, std::vector<Bound>> BoundsMap;

  private:
    const std::unordered_map<const ExprNode *, uint64_t> &hash_; // expr -> hash

    BoundsMap lower_, upper_;

    // buffer table
    std::unordered_map<std::string, Ref<Buffer>> buffers_;
    // iterator table
    std::unordered_map<std::string, std::pair<Expr, Expr>> iters_;

  private:
    std::vector<Bound> getLower(const Expr &op) const;
    std::vector<Bound> getUpper(const Expr &op) const;
    void updLower(const Expr &op, const Bound &bound);
    void updUpper(const Expr &op, const Bound &bound);

    int getIntLower(const Expr &op) const;
    int getIntUpper(const Expr &op) const;
    Ref<int> getInt(const Expr &op) const;

    uint64_t getHash(const Expr &op);

    static Expr sub1(const Expr &op);
    static Expr add1(const Expr &op);

  public:
    AnalyzeBounds(const std::unordered_map<const ExprNode *, uint64_t> &hash)
        : hash_(hash) {}

    const BoundsMap &lower() const { return lower_; }
    const BoundsMap &upper() const { return upper_; }

  protected:
    virtual void visit(const VarDef &op) override;
    virtual void visit(const Var &op) override;
    virtual void visit(const Load &op) override;
    virtual void visit(const IntConst &op) override;
    virtual void visit(const Add &op) override;
    virtual void visit(const Sub &op) override;
    virtual void visit(const Mul &op) override;
    virtual void visit(const Div &op) override;
    virtual void visit(const For &op) override;
    virtual void visit(const If &op) override;
};

} // namespace ir

#endif // BOUNDS_H
