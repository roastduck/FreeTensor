#ifndef FIND_LOOP_VARIANCE_H
#define FIND_LOOP_VARIANCE_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <visitor.h>

namespace ir {

class MarkStores : public Visitor {
    std::vector<std::string> loopStack_;
    std::unordered_map<std::string, std::unordered_set<std::string>>
        &variantVar_;

  public:
    MarkStores(std::unordered_map<std::string, std::unordered_set<std::string>>
                    &variantVar)
        : variantVar_(variantVar) {}

    void visit(const For &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
};

class FindLoopVariance : public Visitor {
    std::vector<std::string> loopStack_;
    std::unordered_map<std::string, std::unordered_set<std::string>>
        variantVar_;
    std::unordered_map<Expr, std::unordered_set<std::string>> variantExpr_;

  public:
    const std::unordered_map<Expr, std::unordered_set<std::string>> &
    variantExpr() const {
        return variantExpr_;
    }

  protected:
    void visit(const For &op) override;
    void visit(const VarDef &op) override;

    template <class T> void visitBinOp(const T &op) {
        Visitor::visit(op);
        if (variantExpr_.count(op->lhs_)) {
            for (auto &&loop : variantExpr_.at(op->lhs_)) {
                variantExpr_[op].insert(loop);
            }
        }
        if (variantExpr_.count(op->rhs_)) {
            for (auto &&loop : variantExpr_.at(op->rhs_)) {
                variantExpr_[op].insert(loop);
            }
        }
    }

    virtual void visit(const Var &op) override;
    virtual void visit(const Load &op) override;
    virtual void visit(const Add &op) override { visitBinOp(op); }
    virtual void visit(const Sub &op) override { visitBinOp(op); }
    virtual void visit(const Mul &op) override { visitBinOp(op); }
    virtual void visit(const RealDiv &op) override { visitBinOp(op); }
    virtual void visit(const FloorDiv &op) override { visitBinOp(op); }
    virtual void visit(const CeilDiv &op) override { visitBinOp(op); }
    virtual void visit(const RoundTowards0Div &op) override { visitBinOp(op); }
    virtual void visit(const Mod &op) override { visitBinOp(op); }
    virtual void visit(const Min &op) override { visitBinOp(op); }
    virtual void visit(const Max &op) override { visitBinOp(op); }
    virtual void visit(const LT &op) override { visitBinOp(op); }
    virtual void visit(const LE &op) override { visitBinOp(op); }
    virtual void visit(const GT &op) override { visitBinOp(op); }
    virtual void visit(const GE &op) override { visitBinOp(op); }
    virtual void visit(const EQ &op) override { visitBinOp(op); }
    virtual void visit(const NE &op) override { visitBinOp(op); }
    virtual void visit(const LAnd &op) override { visitBinOp(op); }
    virtual void visit(const LOr &op) override { visitBinOp(op); }
    virtual void visit(const LNot &op) override;
};

std::unordered_map<Expr, std::unordered_set<std::string>>
findLoopVariance(const AST &op);

} // namespace ir

#endif // FIND_LOOP_VARIANCE_H
