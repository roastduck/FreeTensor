#ifndef CODE_GEN_C_H
#define CODE_GEN_C_H

#include <vector>

#include <pass/code_gen.h>

namespace ir {

class CodeGenC : public CodeGen {
    std::vector<std::string> params_;

  protected:
    virtual void visit(const VarDef &op) override;
    virtual void visit(const Var &op) override;
    virtual void visit(const Store &op) override;
    virtual void visit(const Load &op) override;
    virtual void visit(const IntConst &op) override;
    virtual void visit(const FloatConst &op) override;

  private:
    static std::string gen(DataType dtype);

  public:
    const std::vector<std::string> &params() const { return params_; }
};

/**
 * Generate target function code
 *
 * @return : (source, list of params)
 */
std::pair<std::string, std::vector<std::string>> codeGenC(const AST &op);

} // namespace ir

#endif // CODE_GEN_C_H
