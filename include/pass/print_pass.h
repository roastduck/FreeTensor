#ifndef PRINT_PASS_H
#define PRINT_PASS_H

#include <sstream>
#include <string>

#include <visitor.h>

namespace ir {

class PrintPass : public Visitor {
  protected:
    virtual void visit(const VarDef &op) override;
    virtual void visit(const Var &op) override;
    virtual void visit(const Store &op) override;
    virtual void visit(const Load &op) override;
    virtual void visit(const IntConst &op) override;
    virtual void visit(const FloatConst &op) override;

  private:
    std::ostringstream os;
    int nIndent;

    void makeIndent();

  public:
    std::string toString();
};

} // namespace ir

#endif // PRINT_PASS_H
