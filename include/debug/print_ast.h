#ifndef PRINT_AST_H
#define PRINT_AST_H

#include <unordered_set>

#include <codegen/code_gen.h>

namespace ir {

class PrintVisitor : public CodeGen<CodeGenStream> {
    bool printAllId_ = false, pretty_ = false;
    std::unordered_set<std::string> keywords = {
        "if", "else", "for", "in", "assert", "assume", "func", "true", "false",
    };

  public:
    PrintVisitor(bool printAllId = false, bool pretty = false)
        : printAllId_(printAllId), pretty_(pretty) {}

  private:
    void recur(const Expr &op);
    void recur(const Stmt &op);
    void printId(const Stmt &op);
    std::string printName(const std::string &name);

  protected:
    void visitStmt(const Stmt &op) override;

    void visit(const Func &op) override;
    void visit(const StmtSeq &op) override;
    void visit(const Any &op) override;
    void visit(const AnyExpr &op) override;
    void visit(const VarDef &op) override;
    void visit(const Var &op) override;
    void visit(const Store &op) override;
    void visit(const Load &op) override;
    void visit(const ReduceTo &op) override;
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
    void visit(const For &op) override;
    void visit(const If &op) override;
    void visit(const Assert &op) override;
    void visit(const Assume &op) override;
    void visit(const Intrinsic &op) override;
    void visit(const Eval &op) override;
    void visit(const MatMul &op) override;
};

} // namespace ir

#endif // PRINT_AST_H
