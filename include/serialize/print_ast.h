#ifndef FREE_TENSOR_PRINT_AST_H
#define FREE_TENSOR_PRINT_AST_H

#include <functional>
#include <iostream>
#include <unordered_set>

#include <codegen/code_gen.h>
#include <serialize/stream_utils.h>

namespace freetensor {

class PrintVisitor : public CodeGen<CodeGenStream> {
    bool printAllId_ = false, pretty_ = false, dtypeInLoad_ = false,
         hexFloat_ = false, parenDespitePriority_ = false,
         printSourceLocation_ = false;
    const std::unordered_set<std::string> keywords = {
        "if",     "else",   "for",    "from", "until", "step",
        "length", "assert", "assume", "func", "true",  "false",
    };

    /**
     * @brief Precedence of operators.
     *
     * RHS is for the right hand side of the operator. They require a slightly
     * higher precedence than the operator itself to correctly print parentheses
     * in case of A op (B op C).
     */
    enum class Precedence {
        ANY,
        TRINARY,
        LOR,
        LOR_RHS,
        LAND,
        LAND_RHS,
        COMP,
        COMP_RHS,
        ADD,
        ADD_RHS,
        MUL,
        MUL_RHS,
        UNARY_LOGIC,
    };

    /**
     * @brief Current precedence of the operator.
     *
     * This is used to determine whether to print parentheses.
     */
    Precedence precedence_ = Precedence::ANY;

    /**
     * @brief Enclose the expression printed by inner with correct parentheses.
     *
     * @param new_priority The precedence of the operator in the expression.
     * @param inner The function that prints the expression to enclose.
     * @param parentheses Whether to print parentheses if the precedence is
     * lower than before.
     */
    void precedence_enclose(Precedence new_priority, auto inner,
                            bool parentheses = true) {
        auto old_priority = precedence_;
        precedence_ = new_priority;
        if (parentheses &&
            (parenDespitePriority_ || old_priority > precedence_))
            os() << "(";
        inner();
        if (parentheses &&
            (parenDespitePriority_ || old_priority > precedence_))
            os() << ")";
        precedence_ = old_priority;
    }

    /**
     * @brief Start a new precedence context. Used in enforced parentheses or
     * root expressions.
     *
     * @param inner The function that prints the expression.
     */
    void precedence_new(auto inner) {
        precedence_enclose(Precedence::ANY, inner, false);
    }

  public:
    PrintVisitor(bool printAllId = false, bool pretty = false,
                 bool dtypeInLoad = false, bool hexFloat = false,
                 bool compact = false, bool parenDespitePriority = false,
                 bool printSourceLocation = false)
        : CodeGen(compact), printAllId_(printAllId), pretty_(pretty),
          dtypeInLoad_(dtypeInLoad), hexFloat_(hexFloat),
          parenDespitePriority_(parenDespitePriority),
          printSourceLocation_(printSourceLocation) {
        os() << manipNoIdSign(true)
             << (printSourceLocation ? manipMetadataWithLocation
                                     : manipMetadataSkipLocation);
    }

  private:
    void recur(const Expr &op);
    void recur(const Stmt &op);
    void printMetadataAndId(const Stmt &op);

    std::string escape(const std::string &name);
    std::string prettyIterName(const std::string &name);
    std::string prettyVarDefName(const std::string &name);
    std::string prettyFuncName(const std::string &name);
    std::function<std::ostream &(std::ostream &)> prettyId(const ID &id);
    std::string prettyLiteral(const std::string &lit);
    std::string prettyKeyword(const std::string &kw);
    std::string prettyDType(const DataType &dtype);

  protected:
    void visitStmt(const Stmt &op) override;

    void visit(const Func &op) override;
    void visit(const StmtSeq &op) override;
    void visit(const Any &op) override;
    void visit(const AnyExpr &op) override;
    void visit(const VarDef &op) override;
    void visit(const Var &op) override;
    void visit(const Store &op) override;
    void visit(const Alloc &op) override;
    void visit(const Free &op) override;
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
    void visit(const Ln &op) override;
    void visit(const Square &op) override;
    void visit(const Sigmoid &op) override;
    void visit(const Sin &op) override;
    void visit(const Cos &op) override;
    void visit(const Tan &op) override;
    void visit(const Tanh &op) override;
    void visit(const Abs &op) override;
    void visit(const Floor &op) override;
    void visit(const Ceil &op) override;
    void visit(const Unbound &op) override;
    void visit(const IfExpr &op) override;
    void visit(const Cast &op) override;
    void visit(const For &op) override;
    void visit(const If &op) override;
    void visit(const Assert &op) override;
    void visit(const Assume &op) override;
    void visit(const Intrinsic &op) override;
    void visit(const Eval &op) override;
    void visit(const MatMul &op) override;
    void visit(const MarkVersion &op) override;
    void visit(const LoadAtVersion &op) override;
};

/**
 * Print functions for debugging
 *
 * @{
 */
std::string toString(const AST &op);
std::string toString(const AST &op, bool pretty);
std::string toString(const AST &op, bool pretty, bool printAllId);
std::string toString(const AST &op, bool pretty, bool printAllId,
                     bool dtypeInLoad, bool hexFloat = false,
                     bool compact = false, bool parenDespitePriority = false);
std::string toString(const AST &op, bool pretty, bool printAllId,
                     bool dtypeInLoad, bool hexFloat, bool compact,
                     bool parenDespitePriority, bool printSourceLocation);
/** @} */

/**
 * Serialize function for storing an AST and loading it back
 */
inline std::string dumpAST(const AST &op, bool dtypeInLoad = false,
                           bool hexFloat = true) {
    return toString(op, false, true, dtypeInLoad, hexFloat, true, false, false);
}

/**
 * Control over whether to allow pretty print in a stream
 *
 * Get or set the option via `std::ostream::iword()`, or set it via outputting
 * `manipNoPrettyAST` to the stream
 *
 * If set, this option overrides `Config::prettyPrint()` and force no pretty
 * print. Defaults to unset
 *
 * @{
 */
extern int OSTREAM_NO_PRETTY;
std::function<std::ostream &(std::ostream &)> manipNoPrettyAST(bool flag);
/** @} */

/**
 * Print an AST
 *
 * `OSTREAM_NO_PRETTY` can be set via `manipNoPrettyAST` to enable or disable
 * pretty print for a specific stream
 */
std::ostream &operator<<(std::ostream &os, const AST &op);

} // namespace freetensor

#endif // FREE_TENSOR_PRINT_AST_H
