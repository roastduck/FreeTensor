#ifndef PRINT_AST_H
#define PRINT_AST_H

#include <pass/code_gen.h>

namespace ir {

class PrintVisitor : public CodeGen {
  protected:
    virtual void visit(const Any &op) override;
    virtual void visit(const VarDef &op) override;
    virtual void visit(const Var &op) override;
    virtual void visit(const Store &op) override;
    virtual void visit(const Load &op) override;
    virtual void visit(const AddTo &op) override;
    virtual void visit(const IntConst &op) override;
    virtual void visit(const FloatConst &op) override;
    virtual void visit(const Add &op) override;
    virtual void visit(const Sub &op) override;
    virtual void visit(const Mul &op) override;
    virtual void visit(const Div &op) override;
    virtual void visit(const Mod &op) override;
    virtual void visit(const LT &op) override;
    virtual void visit(const LE &op) override;
    virtual void visit(const GT &op) override;
    virtual void visit(const GE &op) override;
    virtual void visit(const EQ &op) override;
    virtual void visit(const NE &op) override;
    virtual void visit(const Not &op) override;
    virtual void visit(const For &op) override;
    virtual void visit(const If &op) override;
};

std::string printAST(const AST &op);

} // namespace ir

#endif // PRINT_AST_H
