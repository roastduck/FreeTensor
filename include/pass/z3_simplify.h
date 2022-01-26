#ifndef Z3_SIMPLIFY
#define Z3_SIMPLIFY

#include <deque>
#include <unordered_map>

#include <z3++.h>

#include <func.h>
#include <hash.h>
#include <mutator.h>
#include <opt.h>
#include <visitor.h>

namespace ir {
class OutDatedCondsRemover : public Visitor {
    std::deque<std::pair<Expr, bool>> &condList_;

  public:
    OutDatedCondsRemover(std::deque<std::pair<Expr, bool>> &condList)
        : condList_(condList) {}

  private:
    void remove(const std::string &name);

  protected:
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
};

/**
 * Simplify the AST using Z3
 *
 * Comparing to SimplifyPass, Z3Simplify has the following features:
 *
 * - It can only simplify boolean expressions (e.g. remove redundant branches),
 * but it can NOT transform an expression to a simpler form (e.g. transform x +
 * x - x to x)
 * - It can deal with some more complex expressions, such as Mod
 * - It may take some more time
 */
class Z3Simplify : public Mutator {
    int varCnt_ = 0;
    ASTHashMap<Expr, int> varId_;

    z3::context ctx_;
    z3::solver solver_;

    // We use Opt because there is no z3::expr::expr()
    std::unordered_map<Expr, Opt<z3::expr>> z3Exprs_;
    std::deque<std::pair<Expr, bool>> condList_;

    OutDatedCondsRemover remove_;

  public:
    Z3Simplify() : solver_(ctx_), remove_(condList_) {}

  private:
    int getVarId(const Expr &op);

    void put(const Expr &key, const z3::expr &expr);
    bool exists(const Expr &key);
    const z3::expr &get(const Expr &key);

    void push(const Expr &op);
    void pop();

  protected:
    bool prove(const Expr &op);

    using Mutator::visit;

    Expr visit(const Var &op) override;
    Expr visit(const Load &op) override;
    Expr visit(const IntConst &op) override;
    Expr visit(const BoolConst &op) override;
    Expr visit(const Add &op) override;
    Expr visit(const Sub &op) override;
    Expr visit(const Mul &op) override;
    Expr visit(const FloorDiv &op) override;
    Expr visit(const CeilDiv &op) override;
    Expr visit(const Mod &op) override;
    Expr visit(const Min &op) override;
    Expr visit(const Max &op) override;
    Expr visit(const LT &op) override;
    Expr visit(const LE &op) override;
    Expr visit(const GT &op) override;
    Expr visit(const GE &op) override;
    Expr visit(const EQ &op) override;
    Expr visit(const NE &op) override;
    Expr visit(const LAnd &op) override;
    Expr visit(const LOr &op) override;
    Expr visit(const LNot &op) override;
    Expr visit(const IfExpr &op) override;

    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
};

Stmt z3Simplify(const Stmt &op);

DEFINE_PASS_FOR_FUNC(z3Simplify)

} // namespace ir

#endif // Z3_SIMPLIFY
