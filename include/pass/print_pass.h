#ifndef PRINT_PASS_H
#define PRINT_PASS_H

#include <pass/code_gen.h>

namespace ir {

class PrintPass : public CodeGen {
  protected:
    virtual void visit(const VarDef &op) override;
    virtual void visit(const Var &op) override;
    virtual void visit(const Store &op) override;
    virtual void visit(const Load &op) override;
    virtual void visit(const IntConst &op) override;
    virtual void visit(const FloatConst &op) override;
};

std::string printPass(const AST &op);

} // namespace ir

#endif // PRINT_PASS_H
