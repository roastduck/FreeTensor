#ifndef TYPE_INFER_H
#define TYPE_INFER_H

#include <unordered_map>

#include <visitor.h>

namespace ir {

class TypeInfer : public Visitor {
    std::unordered_map<Expr, DataType> types_;
    std::unordered_map<std::string, Ref<Buffer>> *buffers_;
    bool borrowedBuffers_ = false;

  public:
    TypeInfer()
        : buffers_(new std::unordered_map<std::string, Ref<Buffer>>()) {}
    TypeInfer(std::unordered_map<std::string, Ref<Buffer>> *buffers)
        : buffers_(buffers), borrowedBuffers_(true) {}
    ~TypeInfer() {
        if (!borrowedBuffers_) {
            delete buffers_;
        }
    }

    const std::unordered_map<Expr, DataType> types() const { return types_; }

  protected:
    void visitExpr(const Expr &op,
                   const std::function<void(const Expr &)> &visitNode) override;

    void visit(const Var &op) override;
    void visit(const Load &op) override;
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
    void visit(const Intrinsic &op) override;

    void visit(const VarDef &op) override;
};

} // namespace ir

#endif // TYPE_INFER_H
