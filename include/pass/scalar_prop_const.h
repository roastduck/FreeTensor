#ifndef SCALAR_PROP_CONST_H
#define SCALAR_PROP_CONST_H

#include <analyze/symbol_table.h>
#include <func.h>
#include <hash.h>
#include <mutator.h>

#include <map>

namespace ir {

/**
 * Mutator for propagating scalar values that are const or depend on iteration
 * variables only.
 *
 * Scalars are values in tensors indexed with constants, i.e.
 * this pass requires both indices and assigned value to be constants.
 */
class ScalarPropConst : public SymbolTable<Mutator> {
  private:
    typedef SymbolTable<Mutator> BaseClass;
    /**
     * @brief Type dispatch for constant types.
     *
     * @tparam F Functor type for given callback
     * @param c Reference of the constant AST node
     * @param f Callback for processing a typed constant value, should accept
     * any concrete type (through an `auto`/templated parameter)
     * @return auto Returns what `f` returns.
     */
    template <typename F> static auto dispatch(const Const &c, F f) {
        switch (c->nodeType()) {
        case ASTNodeType::IntConst:
            return f(c.as<IntConstNode>()->val_);
        case ASTNodeType::FloatConst:
            return f(c.as<FloatConstNode>()->val_);
        case ASTNodeType::BoolConst:
            return f(c.as<BoolConstNode>()->val_);
        default:
            ASSERT(false && "Unknown Const node");
        }
    }

    /**
     * @brief Wrap a typed value into a constant AST node.
     *
     * @param t Compile-time values to be wrapped
     * @return Const Wrapped `Const` AST node
     * @{
     */
    static Const wrap(const int &t) { return makeIntConst(t).as<ConstNode>(); }

    static Const wrap(const int64_t &t) {
        return makeIntConst(t).as<ConstNode>();
    }

    static Const wrap(const double &t) {
        return makeFloatConst(t).as<ConstNode>();
    }

    static Const wrap(const bool &t) {
        return makeBoolConst(t).as<ConstNode>();
    }
    /** @} */

    /**
     * @brief Cast the data type of a `Const` node.
     *
     * @param type Target type
     * @param val Constant node to be casted
     * @return Const Casted constant node
     */
    static Const castType(DataType type, const Const &val) {
        auto result = dispatch(val, [type](auto v) {
            switch (type) {
            case DataType::Int32:
                return wrap(int64_t(v));
            case DataType::Float32:
            case DataType::Float64:
                return wrap(double(v));
            case DataType::Bool:
                return wrap(bool(v));
            default:
                ASSERT(false && "Unrecognized variable type assigned")
            }
        });
        return COPY_DEBUG_INFO(result, val);
    }

    /**
     * @brief Indices to a scalar, includes a sequence of constant offsets.
     */
    struct ScalarIndices {
        std::vector<int64_t> offset;

        /// Support comparison to use `std::map`.
        bool operator<(const ScalarIndices &other) const {
            ASSERT(offset.size() == other.offset.size() &&
                   "Index count should be identical for same tensor");
            for (size_t i = 0; i < offset.size(); ++i)
                if (offset[i] < other.offset[i])
                    return true;
                else if (offset[i] > other.offset[i])
                    return false;
            return false;
        }

        /// Support equivalence check
        bool operator==(const ScalarIndices &other) const {
            ASSERT(offset.size() == other.offset.size() &&
                   "Index count should be identical for same tensor");
            for (size_t i = 0; i < offset.size(); ++i)
                if (offset[i] != other.offset[i])
                    return false;
            return true;
        }
    };

    /**
     * @brief Try converting indices' AST nodes to constant indices.
     *
     * @param exprs AST nodes for indices
     * @return std::optional<ScalarIndices> Indices to the scalar, if all
     * indices are constant
     */
    std::optional<ScalarIndices>
    tryToScalar(const std::vector<SubTree<ExprNode>> &exprs);

    /// Scalar constants records, with first level map indexing var names and
    /// second indexing indices
    std::unordered_map<std::string, std::map<ScalarIndices, Expr>> constants_;

    /// Constant entries dependent on each iteration variable
    std::unordered_multimap<std::string, std::pair<std::string, ScalarIndices>>
        iter_dep_constants_;

  protected:
    void gen_constant(const std::string &name,
                      const std::optional<ScalarIndices> &indices,
                      const Expr &value);
    void kill_iter_dep_entry(const std::string &name,
                             const ScalarIndices &indices);
    void kill_constant(const std::string &name,
                       const std::optional<ScalarIndices> &indices);
    void kill_iter(const std::string &it_var);

    auto backup_state() { return std::pair{constants_, iter_dep_constants_}; }
    void restore_state(
        std::pair<decltype(constants_), decltype(iter_dep_constants_)> state) {
        std::tie(constants_, iter_dep_constants_) = state;
    }

    /**
     * @brief Intersect currently recorded scalar constants with provided map.
     *
     * This operation removes any record not found in `other` from the current
     * map.
     *
     * @param other The constants map to intersect
     * @return true The current constants are changed by this intersection
     * @return false The current constants remain unchanged in this intersection
     */
    bool intersect_constants_with(
        std::unordered_map<std::string, std::map<ScalarIndices, Expr>> other);

  protected:
    Stmt visit(const Store &store_orig) override;
    Stmt visit(const ReduceTo &op) override;
    Expr visit(const Load &load_orig) override;
    Stmt visit(const If &op) override;
    Stmt visit(const VarDef &vd) override;
    Stmt visit(const For &op) override;

  private:
    /**
     * @brief Generic binary operation visit
     *
     * @tparam F Functor type of the callback
     * @tparam FAlt Functor type for recovering the node
     * @param op The BinaryExpr to visit
     * @param f Callback for constant folding over two statically typed values
     * @param falt Callback for recovering the node
     * @return Expr Result expression, possibly constant-folded
     */
    template <typename F, typename FAlt>
    Expr visitBinary(const BinaryExpr &op, F f, FAlt falt) {
        auto lhs = visitExpr(op->lhs_);
        auto rhs = visitExpr(op->rhs_);
        Expr res;
        if (lhs->isConst() && rhs->isConst())
            res = dispatch(lhs.as<ConstNode>(), [&](auto ll) {
                return dispatch(rhs.as<ConstNode>(),
                                [&](auto rr) { return wrap(f(ll, rr)); });
            });
        else
            res = falt(lhs, rhs);
        return COPY_DEBUG_INFO(res, op);
    }

    /**
     * @brief Generic unary operation visit
     *
     * @tparam F Functor type of the callback
     * @tparam FAlt Functor type for recovering the node
     * @param op The UnaryExpr to visit
     * @param f Callback for constant folding over a statically typed value
     * @param falt Callback for recovering the node
     * @return Expr Result expression, possibly constant-folded
     */
    template <typename F, typename FAlt>
    Expr visitUnary(const UnaryExpr &op, F f, FAlt falt) {
        auto x = visitExpr(op->expr_);
        Expr res;
        if (x->isConst())
            res = dispatch(x.as<ConstNode>(),
                           [&](auto xx) { return wrap(f(xx)); });
        else
            res = falt(x);
        return COPY_DEBUG_INFO(res, op);
    }

  protected:
#define BINARY_OP(OPNAME, OP) Expr visit(const OPNAME &) override;
#define BINARY_OP_F(OPNAME, OPF, OP) Expr visit(const OPNAME &) override;
#define UNARY_OP(OPNAME, OPF) Expr visit(const OPNAME &) override;
    BINARY_OP(Add, +)
    BINARY_OP(Sub, -)
    BINARY_OP(Mul, *)
    BINARY_OP(RealDiv, /)
    BINARY_OP_F(FloorDiv, floorDiv, %)
    BINARY_OP_F(CeilDiv, ceilDiv, %)
    BINARY_OP(RoundTowards0Div, /)
    BINARY_OP_F(Mod, mod, %)
    BINARY_OP(Remainder, %)
    BINARY_OP_F(Min, std::min, +)
    BINARY_OP_F(Max, std::max, +)
    BINARY_OP(LT, <)
    BINARY_OP(LE, <=)
    BINARY_OP(GT, >)
    BINARY_OP(GE, >=)
    BINARY_OP(EQ, ==)
    BINARY_OP(NE, !=)
    BINARY_OP(LAnd, &&)
    BINARY_OP(LOr, ||)
    UNARY_OP(LNot, !)
    UNARY_OP(Sqrt, std::sqrt)
    UNARY_OP(Exp, std::exp)

  private:
    static int64_t _square(const int64_t &t) { return t * t; }
    static double _square(const double &t) { return t * t; }

  protected:
    UNARY_OP(Square, _square)
    //! TODO: Sigmoid
    //! TODO: Tanh
    UNARY_OP(Abs, std::abs)
    UNARY_OP(Floor, std::floor)
    UNARY_OP(Ceil, std::ceil)
#undef BINARY_OP
#undef BINARY_OP_F
#undef UNARY_OP

    Expr visit(const Cast &op) override;
}; // namespace ir

/**
 * Propagate scalars of constant value or only depending on iteration variables.
 * Scalars are values in tensors indexed with constants.
 *
 * E.g. transform
 *
 * ```
 * x[0] = 1
 * y[0] = x[0]
 * ```
 *
 * into
 *
 * ```
 * x[0] = 1
 * y[0] = 1
 * ```
 *
 * This version of const propagation is designed for only scalars and meant to
 * be fast. It uses traditional dataflow techniques
 */
Stmt scalarPropConst(const Stmt &op);

DEFINE_PASS_FOR_FUNC(scalarPropConst)

} // namespace ir

#endif
