#ifndef FREE_TENSOR_FIND_LOOP_VARIANCE_H
#define FREE_TENSOR_FIND_LOOP_VARIANCE_H

#include <unordered_map>
#include <vector>

#include <analyze/symbol_table.h>
#include <analyze/track_stmt.h>
#include <visitor.h>

namespace freetensor {

enum class LoopVariability : int {
    // Non-existence = Invariant
    Unknown,
    Variant,
};

typedef std::unordered_map<std::string, std::unordered_map<ID, LoopVariability>>
    LoopVariTransVarMap;
typedef std::unordered_map<ID /* of VarDef */,
                           std::unordered_map<ID, LoopVariability>>
    LoopVariUniqVarMap;
typedef std::unordered_map<StmtOrExprID /* of Expr */,
                           std::unordered_map<ID, LoopVariability>>
    LoopVariExprMap;

/**
 * Initialize all expressions to Unkown to its surrounding loops
 */
class InitExprVari : public TrackStmt<Visitor> {
    typedef TrackStmt<Visitor> BaseClass;

    LoopVariExprMap &exprInfo_;
    std::vector<ID> loopStack_;

  public:
    InitExprVari(LoopVariExprMap &exprInfo) : exprInfo_(exprInfo) {}

  protected:
    void visitExpr(const Expr &expr) override;
    void visit(const For &op) override;
};

/**
 * Determine variance of a variable by expressions
 *
 * Variance of this particular variable is initialized to Invariant, while
 * variance of expressions are kept as-is. Therefore:
 *
 * - If any of the expressions stored to it is Variant, it will become Variant
 * - If no expressions stored to it is Variant, but some of them are Unknown, it
 * will become Unknown
 * - If all of the expressions stored to it is Invariant, it will become
 * Invariant
 */
class MarkStores : public TrackStmt<Visitor> {
    typedef TrackStmt<Visitor> BaseClass;

    const std::string &var_;
    std::vector<StmtOrExprID> &condStack_;
    LoopVariTransVarMap &varInfo_;
    const LoopVariExprMap &exprInfo_;

  public:
    MarkStores(const std::string &var, std::vector<StmtOrExprID> &condStack,
               LoopVariTransVarMap &varInfo, const LoopVariExprMap &exprInfo)
        : var_(var), condStack_(condStack), varInfo_(varInfo),
          exprInfo_(exprInfo) {
        varInfo_.erase(var_); // Invariant
    }

  private:
    // to = from meet to
    void meetTo(const Expr &from, const std::string &to);
    void meetTo(const StmtOrExprID &from, const std::string &to);

    template <class T> void visitMemWrite(const T &op) {
        BaseClass::visit(op);
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
    void visit(const ReduceTo &op) override {
        visitMemWrite(op);
        for (auto p = op->parentStmt(); p.isValid(); p = p->parentStmt()) {
            if (p->nodeType() == ASTNodeType::VarDef &&
                p.as<VarDefNode>()->name_ == op->var_) {
                break;
            }
            if (p->nodeType() == ASTNodeType::For) {
                varInfo_[op->var_][p->id()] = LoopVariability::Variant;
            }
        }
    }
};

class FindLoopVariance : public SymbolTable<TrackStmt<Visitor>> {
    typedef SymbolTable<TrackStmt<Visitor>> BaseClass;

    std::vector<ID> loopStack_;
    std::vector<StmtOrExprID> condStack_;
    LoopVariTransVarMap varInfo_;
    LoopVariUniqVarMap uniqVarInfo_;
    LoopVariExprMap exprInfo_;

  public:
    FindLoopVariance(const Stmt &root) { InitExprVari{exprInfo_}(root); }

    const LoopVariExprMap &exprInfo() const { return exprInfo_; }
    const LoopVariUniqVarMap &varInfo() const { return uniqVarInfo_; }

    int unknownCnt() const;

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
    void visit(const Intrinsic &op) override;
};

bool isVariant(const LoopVariExprMap &exprInfo, const StmtOrExprID &expr,
               const ID &loop);
bool isVariant(const LoopVariUniqVarMap &varInfo, const VarDef &def,
               const ID &loop);
bool isVariant(const LoopVariUniqVarMap &varInfo, const ID &defId,
               const ID &loop);

/**
 * Check whether an expression or a variable is loop-variant
 *
 * This function returns two map. The first map shows whether an expression is
 * loop-variant, while the second map shows whether a variable is loop-variant.
 * The result should be get by calling `isVariant`
 *
 * NOTE: `findLoopVariance` currently treat expressions invariant to all their
 * non-surrounding loops for minimize analyzing overhead, but it is possible to
 * make this behaviour optional (see FindLoopVariance::visit(const Load &))
 *
 * `findLoopVariance` runs an iterative algorithm. The variance info is
 * expressed as a semi-lattice:
 *
 * ```
 *   Invariant (implemented by inexistence in the resulting map)
 *       |
 *    Unknown  (implemented by the `LoopVariability` type)
 *       |
 *    Variant  (implemented by the `LoopVariability` type)
 * ```
 *
 * All variabilities are initialized to Unkown, and will become either Invariant
 * or Variant during iterations. The variability of an expression is the "meet"
 * of its sub-expressions' variability. The variability of a variable is the
 * "meet" of variability of all expressions stored to it
 *
 * It is suggested to call pass/simplify before this function, to avoid false
 * variants like `i - i`
 */
std::pair<LoopVariExprMap, LoopVariUniqVarMap> findLoopVariance(const Stmt &op);

} // namespace freetensor

#endif // FREE_TENSOR_FIND_LOOP_VARIANCE_H
