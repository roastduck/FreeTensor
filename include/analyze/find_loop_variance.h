#ifndef FIND_LOOP_VARIANCE_H
#define FIND_LOOP_VARIANCE_H

#include <unordered_map>
#include <vector>

#include <visitor.h>

namespace ir {

enum class LoopVariability : int { Variance, Invariance };

typedef std::unordered_map<std::string,
                           std::unordered_map<std::string, LoopVariability>>
    LoopVariTransVarMap;
typedef std::unordered_map<VarDef,
                           std::unordered_map<std::string, LoopVariability>>
    LoopVariUniqVarMap;
typedef std::unordered_map<Expr,
                           std::unordered_map<std::string, LoopVariability>>
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
            varInfo_[var_][loop->id()] = LoopVariability::Invariance;
        }
    }

  private:
    void mergeInfo(const Expr &from, const std::string &to);

    template <class T> void visitMemWrite(const T &op) {
        Visitor::visit(op);
        if (op->var_ == var_) {
            mergeInfo(op->expr_, op->var_);
            for (auto &&index : op->indices_) {
                mergeInfo(index, op->var_);
            }
            for (auto &&cond : condStack_) {
                mergeInfo(cond, op->var_);
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
    const std::vector<std::string> &allLoops_;

    std::vector<For> loopStack_;
    std::vector<Expr> condStack_;
    LoopVariTransVarMap varInfo_;
    LoopVariUniqVarMap uniqVarInfo_;
    LoopVariExprMap exprInfo_;

  public:
    FindLoopVariance(const std::vector<std::string> &allLoops)
        : allLoops_(allLoops) {}

    const LoopVariExprMap &exprInfo() const { return exprInfo_; }
    const LoopVariUniqVarMap &varInfo() const { return uniqVarInfo_; }

    int knownCnt() const;

  private:
    void copyInfo(const Expr &from, const Expr &to);
    void mergeInfo(const Expr &from, const Expr &to);

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
               const std::string &loop);
bool isVariant(const LoopVariUniqVarMap &varInfo, const VarDef &def,
               const std::string &loop);

std::pair<LoopVariExprMap, LoopVariUniqVarMap> findLoopVariance(const Stmt &op);

} // namespace ir

#endif // FIND_LOOP_VARIANCE_H
