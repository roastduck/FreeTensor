#ifndef FREE_TENSOR_CODE_GEN_C_H
#define FREE_TENSOR_CODE_GEN_C_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <codegen/code_gen.h>

namespace freetensor {

template <class Stream> class CodeGenC : public CodeGen<Stream> {
    typedef CodeGen<Stream> BaseClass;

    const std::vector<FuncParam> &params_;
    const std::vector<FuncRet> &returns_;

  public:
    CodeGenC(const std::vector<FuncParam> &params,
             const std::vector<FuncRet> &returns)
        : params_(params), returns_(returns) {}

    static std::string gen(DataType dtype);

  protected:
    virtual void genAlloc(const Ref<Tensor> &tensor, const std::string &rawPtr,
                          const std::string &shapePtr,
                          const std::string &dimPtr) = 0;

    // Generate the access to a scalar or an element of an array
    virtual void genScalar(const std::string &var,
                           const std::vector<Expr> &indices);
    template <class T> void genScalar(const T &op) {
        genScalar(op->var_, op->indices_);
    }

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
    virtual void visit(const Remainder &op) override;
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
    virtual void visit(const Sigmoid &op) override;
    virtual void visit(const Tanh &op) override;
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

} // namespace freetensor

#endif // FREE_TENSOR_CODE_GEN_C_H
