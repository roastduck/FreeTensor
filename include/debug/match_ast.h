#ifndef MATCH_AST_H
#define MATCH_AST_H

#include <unordered_map>

#include <visitor.h>

namespace ir {

/**
 * Check whether an AST strictly matches a pattern
 *
 * Note that a + b doesn't match b + a
 */
class MatchVisitor : public Visitor {
    bool isMatched_ = true;
    AST instance_;

    std::unordered_map<std::string, std::string> nameMap_;

  public:
    MatchVisitor(const AST &instance) : instance_(instance) {}

    bool isMatched() const { return isMatched_; }

    bool matchName(const std::string &thisName, const std::string &otherName);

  protected:
    virtual void visit(const StmtSeq &op) override;
    virtual void visit(const VarDef &op) override;
    virtual void visit(const Var &op) override;
    virtual void visit(const Store &op) override;
    virtual void visit(const Load &op) override;
    virtual void visit(const ReduceTo &op) override;
    virtual void visit(const IntConst &op) override;
    virtual void visit(const FloatConst &op) override;
    virtual void visit(const Add &op) override;
    virtual void visit(const Sub &op) override;
    virtual void visit(const Mul &op) override;
    virtual void visit(const RealDiv &op) override;
    virtual void visit(const FloorDiv &op) override;
    virtual void visit(const CeilDiv &op) override;
    virtual void visit(const RoundTowards0Div &op) override;
    virtual void visit(const Mod &op) override;
    virtual void visit(const Min &op) override;
    virtual void visit(const Max &op) override;
    virtual void visit(const LT &op) override;
    virtual void visit(const LE &op) override;
    virtual void visit(const GT &op) override;
    virtual void visit(const GE &op) override;
    virtual void visit(const EQ &op) override;
    virtual void visit(const NE &op) override;
    virtual void visit(const LAnd &op) override;
    virtual void visit(const LOr &op) override;
    virtual void visit(const LNot &op) override;
    virtual void visit(const For &op) override;
    virtual void visit(const If &op) override;
    virtual void visit(const Assert &op) override;
    virtual void visit(const Intrinsic &op) override;
    virtual void visit(const Eval &op) override;
};

} // namespace ir

#endif // MATCH_AST_H
