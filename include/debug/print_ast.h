#ifndef PRINT_AST_H
#define PRINT_AST_H

#include <codegen/code_gen.h>

namespace ir {

class PrintVisitor : public CodeGen {
  private:
    void recur(const Expr &op);
    void recur(const Stmt &op);
    void printId(const Stmt &op);

  protected:
    virtual void visit(const Func &op) override;
    virtual void visit(const Any &op) override;
    virtual void visit(const AnyExpr &op) override;
    virtual void visit(const VarDef &op) override;
    virtual void visit(const Var &op) override;
    virtual void visit(const Store &op) override;
    virtual void visit(const Load &op) override;
    virtual void visit(const ReduceTo &op) override;
    virtual void visit(const IntConst &op) override;
    virtual void visit(const FloatConst &op) override;
    virtual void visit(const BoolConst &op) override;
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

std::string printAST(const AST &op);

} // namespace ir

#endif // PRINT_AST_H
