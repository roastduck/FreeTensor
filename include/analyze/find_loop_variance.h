#ifndef FREE_TENSOR_FIND_LOOP_VARIANCE_H
#define FREE_TENSOR_FIND_LOOP_VARIANCE_H

#include <unordered_map>
#include <vector>

#include <visitor.h>

namespace freetensor {

enum class LoopVariability : int { Variant, Invariant };

typedef std::unordered_map<std::string, std::unordered_map<ID, LoopVariability>>
    LoopVariTransVarMap;
typedef std::unordered_map<VarDef, std::unordered_map<ID, LoopVariability>>
    LoopVariUniqVarMap;
typedef std::unordered_map<Expr, std::unordered_map<ID, LoopVariability>>
    LoopVariExprMap;

class MarkStores : public Visitor {
    const std::string &var_;
    std::vector<For> &loopStack_;
    std::vector<Expr> &condStack_;
    LoopVariTransVarMap &varInfo_;
    const LoopVariExprMap &exprInfo_;

  public:
    MarkStores(const std::string &var, std::vector<For> &loopStack,
               std::vector<Expr> &condStack, LoopVariTransVarMap &varInfo,
               const LoopVariExprMap &exprInfo)
        : var_(var), loopStack_(loopStack), condStack_(condStack),
          varInfo_(varInfo), exprInfo_(exprInfo) {
        for (auto &&loop : loopStack_) {
            varInfo_[var_][loop->id()] = LoopVariability::Invariant;
        }
    }

  private:
    // to = from meet to
    void meetTo(const Expr &from, const std::string &to);

    template <class T> void visitMemWrite(const T &op) {
        Visitor::visit(op);
        if (op->var_ == var_) {
            meetTo(op->expr_, op->var_);
            for (auto &&index : op->indices_) {
                meetTo(index, op->var_);
            }
            for (auto &&cond : condStack_) {
                meetTo(cond, op->var_);
            }
        }
    }

  protected:
    void visit(const For &op) override;
    void visit(const If &op) override;
    void visit(const Store &op) override { visitMemWrite(op); }
    void visit(const ReduceTo &op) override { visitMemWrite(op); }
};

class FindLoopVariance : public Visitor {
    const std::vector<ID> &allLoops_;

    std::vector<For> loopStack_;
    std::vector<Expr> condStack_;
    LoopVariTransVarMap varInfo_;
    LoopVariUniqVarMap uniqVarInfo_;
    LoopVariExprMap exprInfo_;

  public:
    FindLoopVariance(const std::vector<ID> &allLoops) : allLoops_(allLoops) {}

    const LoopVariExprMap &exprInfo() const { return exprInfo_; }
    const LoopVariUniqVarMap &varInfo() const { return uniqVarInfo_; }

    int knownCnt() const;

  private:
    // to = from
    void copyInfo(const Expr &from, const Expr &to);
    // to = from meet to
    void meetTo(const Expr &from, const Expr &to);

    void visitConst(const Const &op);
    void visitBinOp(const BinaryExpr &op);
    void visitUnaryOp(const UnaryExpr &op);

  protected:
    void visit(const For &op) override;
    void visit(const If &op) override;
    void visit(const VarDef &op) override;

    void visitExpr(const Expr &op) override;
    void visit(const Var &op) override;
    void visit(const Load &op) override;
    void visit(const IfExpr &op) override;
    void visit(const Cast &op) override;
};

bool isVariant(const LoopVariExprMap &exprInfo, const Expr &expr,
               const ID &loop);
bool isVariant(const LoopVariUniqVarMap &varInfo, const VarDef &def,
               const ID &loop);

/**
 * Check whether an expression or a variable is loop-variant
 *
 * This function returns two map. The first map shows whether an expression is
 * loop-variant, while the second map shows whether a variable is loop-variant.
 * The result should be get by calling `isVariant`
 *
 * `findLoopVariance` runs an iterative algorithm. The variance info is
 * expressed as a semi-lattice:
 *
 * ```
 *   Invariant
 *       |
 *    Unknown
 *       |
 *    Variant
 * ```
 *
 * All variables are initialized to Unkown (implemented by inexistence in the
 * resulting map), and will become either Invariant or Variant (implemented by
 * the `LoopVariability` type) during iterations. The variability of an
 * expression is the "meet" of its sub-expressions' variability. The variability
 * of a variable is the "meet" of variability of all expressions stored to it
 */
std::pair<LoopVariExprMap, LoopVariUniqVarMap> findLoopVariance(const Stmt &op);

} // namespace freetensor

#endif // FREE_TENSOR_FIND_LOOP_VARIANCE_H
