#ifndef CODE_GEN_C_H
#define CODE_GEN_C_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <codegen/code_gen.h>

namespace ir {

class CodeGenC : public CodeGen {
    std::vector<std::string> params_;
    std::unordered_map<std::string, std::string> idCache_; // IR IDs -> C IDs
    std::unordered_set<std::string> idFlag_;               // C IDs

  protected:
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
    virtual void visit(const Min &op) override;
    virtual void visit(const Max &op) override;
    virtual void visit(const LT &op) override;
    virtual void visit(const LE &op) override;
    virtual void visit(const GT &op) override;
    virtual void visit(const GE &op) override;
    virtual void visit(const EQ &op) override;
    virtual void visit(const NE &op) override;
    virtual void visit(const Not &op) override;
    virtual void visit(const For &op) override;
    virtual void visit(const If &op) override;
    virtual void visit(const Assert &op) override;

  private:
    const std::string &normalizeId(const std::string &id);
    static std::string gen(DataType dtype);

  public:
    const std::vector<std::string> &params() const { return params_; }
};

} // namespace ir

#endif // CODE_GEN_C_H
