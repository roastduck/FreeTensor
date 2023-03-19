#ifndef FREE_TENSOR_DERIVATIVE_H
#define FREE_TENSOR_DERIVATIVE_H

#include <optional>
#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <mutator.h>
#include <visitor.h>

namespace freetensor {

/**
 * Find derivative of each expression
 *
 * Gradients will be updated by multiplying the derivatives
 *
 * `Derivative` has two phase. In the first phase, derivative expressions are
 * built according to mathematicall principles, but the variables they load may
 * not exist in an actual backward pass. The result is stored in
 * `Derivative::LazyFullDerivative`.
 *
 * The result of the first phase can be used to decide which variabels have to
 * be saved to tape or recomputed.
 *
 * In the second phase, given the information of tape and recomputation, the
 * variables in derivative expressions are corrected.
 */
class Derivative : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

  public:
    /**
     * Lazy partial derivative dy/dx_i
     */
    class LazyPartialDerivative {
        SymbolTableData symbolTable_;
        Expr mathExpr_;
        ID rootStmtID_;

        // Additional info to replace `y = f(x)`'s derivative to use `y`. The
        // replacement has to be done lazily, because the version of `y` is
        // different from version of `x`
        Store rootStore_;
        std::optional<bool> usingStore_;

      public:
        LazyPartialDerivative() {} // Unintialized

        LazyPartialDerivative(const SymbolTableData &symbolTable,
                              const Expr &mathExpr, const ID &rootStmtID,
                              const Store &rootStore = nullptr)
            : symbolTable_(symbolTable), mathExpr_(mathExpr),
              rootStmtID_(rootStmtID), rootStore_(rootStore) {}

        /// += another partial derivative
        void merge(const LazyPartialDerivative &other) {
            ASSERT(other.rootStmtID_ == rootStmtID_);
            ASSERT(other.rootStore_ == rootStore_);
            mathExpr_ = makeAdd(mathExpr_, other.mathExpr_);
            usingStore_ = std::nullopt;
        }

        /// Raw derivative expression. `Load`s in the statements loads forward
        /// variables named in the original program, not the final backward pass
        const Expr &mathExpr() const { return mathExpr_; }

        /// True if using `y` for `y = f(x)`'s derivative
        bool usingStore();

        /// Replace any expression in the same context with this
        /// LazyPartialDerivative to use taped or recomputed variables
        Expr
        replaceExpr(const std::unordered_map<ID, std::string> &intermediatesMap,
                    const std::unordered_map<StmtOrExprID, Expr> &versions,
                    const Expr &expr);

        /// Generate finaly derivative expression, where all `Load` loads from
        /// actual tape or recomputation
        Expr
        genReplaced(const std::unordered_map<ID, std::string> &intermediatesMap,
                    const std::unordered_map<StmtOrExprID, Expr> &versions) {
            return replaceExpr(intermediatesMap, versions, mathExpr_);
        }
    };

    /**
     * Lazy full derivative dy/dx_i for all x_i that continuously infuences y
     *
     * Notes on non-differential cases:
     *
     * - If some x_i is missing from a LazyFullDerivative, it means x_i is not
     * continuously infuences y, for example when x_i is an integer. So we can
     * safely ignore all missing items.
     *
     * - If error_ is set, it means there is some errors or unsupported cases.
     * It will be re-thrown if calling genGrads.
     */
    class LazyFullDerivative {
        // (x, dy/dx). Instead of using a map, an ordered list is used to ensure
        // that the statements generated by `genGrads` are consistently ordered
        std::vector<std::pair<Load, LazyPartialDerivative>> partials_;

        // Union of all partials. This is a lazily cached variable
        std::optional<bool> usingStore_;

        std::exception_ptr error_;

      public:
        void addPartial(const Load &x, const LazyPartialDerivative &partial);

        void setError(const std::exception_ptr &error) { error_ = error; }

        bool usingStore();

        std::vector<Stmt>
        genGrads(const std::unordered_map<ID, std::string> &intermediatesMap,
                 const std::unordered_map<StmtOrExprID, Expr> &versions,
                 const std::unordered_map<std::string, std::string> &gradNames,
                 const Expr &gradY);
    };

  private:
    std::unordered_map<StmtOrExprID, LazyFullDerivative> derivatives_;
    std::unordered_map<Expr, LazyPartialDerivative> partials_;
    StmtOrExprID rootExpr_;
    Store rootStore_;

    void setPartial(const Expr &expr, const Expr &partial);

  public:
    const auto &derivatives() const { return derivatives_; }

  protected:
    using BaseClass::visit;

    void visitExpr(const Expr &expr) override;

    // If we have `y = f(x)` as a `Store` node, we can use `y` in the
    // derivative. Please note that this does not apply to `ReduceTo` nodes: in
    // `y += exp(x)`, we have no variable equals to `exp(x)`
    void visit(const Store &op) override;

    void visit(const Load &op) override;
    void visit(const Add &op) override;
    void visit(const Sub &op) override;
    void visit(const Mul &op) override;
    void visit(const RealDiv &op) override;
    void visit(const Min &op) override;
    void visit(const Max &op) override;
    void visit(const IfExpr &op) override;
    void visit(const Sqrt &op) override;
    void visit(const Exp &op) override;
    void visit(const Ln &op) override;
    void visit(const Square &op) override;
    void visit(const Sigmoid &op) override;
    void visit(const Tanh &op) override;
    void visit(const Abs &op) override;
    void visit(const Intrinsic &op) override;
};

} // namespace freetensor

#endif // FREE_TENSOR_DERIVATIVE_H
