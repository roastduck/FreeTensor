#ifndef CODE_GEN_C_H
#define CODE_GEN_C_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <analyze/type_infer.h>
#include <codegen/code_gen.h>

namespace ir {

template <class Stream> class CodeGenC : public CodeGen<Stream> {
    const std::vector<std::string> &params_;
    std::unordered_map<std::string, std::string> idCache_; // IR IDs -> C IDs
    std::unordered_set<std::string> idFlag_;               // C IDs
    TypeInfer typeInfer_;

  public:
    CodeGenC(const std::vector<std::string> &params)
        : params_(params), typeInfer_(&this->buffers_) {}

    const std::string &normalizeId(const std::string &id);

    static std::string gen(DataType dtype);

  protected:
    DataType dtype(const Expr &op);

    virtual void visit(const StmtSeq &op) override;
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
    virtual void visit(const Sqrt &op) override;
    virtual void visit(const Exp &op) override;
    virtual void visit(const Square &op) override;
    virtual void visit(const Abs &op) override;
    virtual void visit(const Floor &op) override;
    virtual void visit(const Ceil &op) override;
    virtual void visit(const IfExpr &op) override;
    virtual void visit(const Cast &op) override;
    virtual void visit(const For &op) override;
    virtual void visit(const If &op) override;
    virtual void visit(const Assert &op) override;
    virtual void visit(const Intrinsic &op) override;
    virtual void visit(const Eval &op) override;
};

} // namespace ir

#endif // CODE_GEN_C_H
