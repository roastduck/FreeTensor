#ifndef MATCH_AST_H
#define MATCH_AST_H

#include <unordered_map>
#include <unordered_set>

#include <visitor.h>

namespace ir {

/**
 * Check whether an AST strictly matches a pattern
 *
 * Names of the variables can be different between the two ASTs, but there must
 * be an one-to-one mapping
 *
 * MatchVisitor can tolerate some difference such as a + b will match b + a, but
 * more complex ones such as (a - b) + c does not match a - (b - c)
 */
class MatchVisitor : public Visitor {
    bool isMatched_ = true;
    AST instance_;

    std::unordered_map<std::string, std::string> nameMap_;
    std::unordered_set<std::string> nameMapImage_;

  public:
    MatchVisitor(const AST &instance) : instance_(instance) {}

    bool isMatched() const { return isMatched_; }

    bool matchName(const std::string &thisName, const std::string &otherName);
    void clearName(const std::string &thisName);

  protected:
    void visit(const StmtSeq &op) override;
    void visit(const VarDef &op) override;
    void visit(const Var &op) override;
    void visit(const Store &op) override;
    void visit(const Load &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const IntConst &op) override;
    void visit(const FloatConst &op) override;
    void visit(const BoolConst &op) override;
    void visit(const Add &op) override;
    void visit(const Sub &op) override;
    void visit(const Mul &op) override;
    void visit(const RealDiv &op) override;
    void visit(const FloorDiv &op) override;
    void visit(const CeilDiv &op) override;
    void visit(const RoundTowards0Div &op) override;
    void visit(const Mod &op) override;
    void visit(const Remainder &op) override;
    void visit(const Min &op) override;
    void visit(const Max &op) override;
    void visit(const LT &op) override;
    void visit(const LE &op) override;
    void visit(const GT &op) override;
    void visit(const GE &op) override;
    void visit(const EQ &op) override;
    void visit(const NE &op) override;
    void visit(const LAnd &op) override;
    void visit(const LOr &op) override;
    void visit(const LNot &op) override;
    void visit(const Sqrt &op) override;
    void visit(const Exp &op) override;
    void visit(const Square &op) override;
    void visit(const Sigmoid &op) override;
    void visit(const Tanh &op) override;
    void visit(const Abs &op) override;
    void visit(const Floor &op) override;
    void visit(const Ceil &op) override;
    void visit(const IfExpr &op) override;
    void visit(const Cast &op) override;
    void visit(const For &op) override;
    void visit(const If &op) override;
    void visit(const Assert &op) override;
    void visit(const Intrinsic &op) override;
    void visit(const Eval &op) override;
    void visit(const MatMul &op) override;
};

} // namespace ir

#endif // MATCH_AST_H
