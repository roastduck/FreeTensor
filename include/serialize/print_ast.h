#ifndef FREE_TENSOR_PRINT_AST_H
#define FREE_TENSOR_PRINT_AST_H

#include <unordered_set>

#include <codegen/code_gen.h>

namespace freetensor {

class PrintVisitor : public CodeGen<CodeGenStream> {
    bool printAllId_ = false, pretty_ = false, dtypeInLoad_ = false;
    const std::unordered_set<std::string> keywords = {
        "if", "else", "for", "in", "assert", "assume", "func", "true", "false",
    };

    enum class Priority {
        ANY,
        TRINARY,
        BINARY_LOGIC,
        COMP,
        ADD,
        MUL,
        UNARY_LOGIC,
    } priority_ = Priority::ANY;

    void priority_enclose(Priority new_priority, auto inner) {
        auto old_priority = priority_;
        priority_ = new_priority;
        if (old_priority > priority_)
            os() << "(";
        inner();
        if (old_priority > priority_)
            os() << ")";
        priority_ = old_priority;
    }

    void priority_new(auto inner, Priority new_priority = Priority::ANY) {
        auto old_priority = priority_;
        priority_ = new_priority;
        inner();
        priority_ = old_priority;
    }

  public:
    PrintVisitor(bool printAllId = false, bool pretty = false,
                 bool dtypeInLoad = false)
        : printAllId_(printAllId), pretty_(pretty), dtypeInLoad_(dtypeInLoad) {}

  private:
    void recur(const Expr &op);
    void recur(const Stmt &op);
    void printId(const Stmt &op);

    std::string escape(const std::string &name);
    std::string prettyIterName(const std::string &name);
    std::string prettyVarDefName(const std::string &name);
    std::string prettyFuncName(const std::string &name);
    std::string prettyId(const std::string &id);
    std::string prettyLiteral(const std::string &lit);
    std::string prettyKeyword(const std::string &kw);

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

// Print functions for debugging
std::string toString(const AST &op);
std::string toString(const AST &op, bool pretty);
std::string toString(const AST &op, bool pretty, bool printAllId);
std::string toString(const AST &op, bool pretty, bool printAllId,
                     bool dtypeInLoad);

// Serialize function for storing an AST and loading it back
inline std::string dumpAST(const AST &op, bool dtypeInLoad = false) {
    return toString(op, false, false, dtypeInLoad);
}

} // namespace freetensor

#endif // FREE_TENSOR_PRINT_AST_H
