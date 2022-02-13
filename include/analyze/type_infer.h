#ifndef TYPE_INFER_H
#define TYPE_INFER_H

#include <unordered_map>

#include <analyze/symbol_table.h>
#include <visitor.h>

namespace ir {

/**
 * Infer the data type of each (sub)expressions in an AST
 *
 * Pass it a pointer to a symbol table to use
 */
class TypeInfer : public Visitor {
    std::unordered_map<Expr, DataType> types_;
    const SymbolTableInterface &symbolTable_;

  public:
    TypeInfer(const SymbolTableInterface &symbolTable)
        : symbolTable_(symbolTable) {}

    const std::unordered_map<Expr, DataType> &types() const { return types_; }

  protected:
    void visitExpr(const Expr &op) override;

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
    void visit(const Remainder &op) override;
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
    void visit(const Square &op) override;
    void visit(const Sigmoid &op) override;
    void visit(const Tanh &op) override;
    void visit(const Abs &op) override;
    void visit(const Floor &op) override;
    void visit(const Ceil &op) override;
    void visit(const IfExpr &op) override;
    void visit(const Cast &op) override;
    void visit(const Intrinsic &op) override;

    void visit(const VarDef &op) override;
};

/**
 * A helper class to invoke type inference inside a Visitor or Mutator
 *
 * Inherit this class to use. The BaseClass should be a decent of
 * SymbolTable<...>
 */
template <class BaseClass> class WithTypeInfer : public BaseClass {
    TypeInfer typeInfer_;

  protected:
    WithTypeInfer() : typeInfer_(*this) {}

    DataType dtype(const Expr &op) {
        typeInfer_(op);
        return typeInfer_.types().at(op);
    }
};

} // namespace ir

#endif // TYPE_INFER_H
