
#include <analyze/all_iters.h>
#include <analyze/all_reads.h>

#include <pass/make_reduction.h>
#include <pass/scalar_prop_const.h>
#include <pass/undo_make_reduction.h>

#include <math/utils.h>

#include <hash.h>
#include <mutator.h>

#include <cmath>
#include <map>
#include <stack>

namespace ir {

/**
 * Mutator for propagating scalar values that are const or depend on iteration
 * variables only.
 *
 * Scalars are values in tensors indexed with constants, i.e.
 * this pass requires both indices and assigned value to be constants.
 */
class ScalarPropConst : public Mutator {
  private:
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
    tryToScalar(const std::vector<SubTree<ExprNode>> &exprs) {
        ScalarIndices res;
        for (auto &i : exprs)
            if (i->nodeType() == ASTNodeType::IntConst)
                res.offset.push_back(i.as<IntConstNode>()->val_);
            else
                return std::nullopt;
        return std::move(res);
    }

    /// Scalar constants records, with first level map indexing var names and
    /// second indexing indices
    std::unordered_map<std::string, std::map<ScalarIndices, Expr>> constants_;

    /// Type of currently available `vardef`s
    std::unordered_map<std::string, DataType> tensors_type_;

    /// Constant entries dependent on each iteration variable
    std::unordered_multimap<std::string, std::pair<std::string, ScalarIndices>>
        iter_dep_constants_;

    static std::optional<std::unordered_set<std::string>>
    try_all_iters(const Expr &expr) {
        if (!allReads(expr).empty())
            return std::nullopt;
        return allIters(expr);
    }

    void gen_constant(const std::string &name,
                      const std::optional<ScalarIndices> &indices,
                      const Expr &value) {
        kill_constant(name, indices);
        if (!indices || !allReads(value).empty())
            return;
        constants_[name][*indices] = value;
        for (auto &it_var : allIters(value)) {
            iter_dep_constants_.insert({it_var, {name, *indices}});
        }
    }

    void kill_iter_dep_entry(const std::string &name,
                             const ScalarIndices &indices) {
        for (auto &it_var : allIters(constants_[name][indices])) {
            auto [range_begin, range_end] =
                iter_dep_constants_.equal_range(it_var);
            for (auto it_const = range_begin; it_const != range_end;) {
                auto prev_it_const = it_const++;
                if (prev_it_const->second.first == name &&
                    prev_it_const->second.second == indices) {
                    iter_dep_constants_.erase(prev_it_const);
                }
            }
        }
    }

    void kill_constant(const std::string &name,
                       const std::optional<ScalarIndices> &indices) {
        if (indices) {
            if (constants_[name].count(*indices))
                kill_iter_dep_entry(name, *indices);
            constants_[name].erase(*indices);
        } else {
            for (const auto &[killing_indices, _] : constants_[name]) {
                kill_iter_dep_entry(name, killing_indices);
            }
            constants_[name].clear();
        }
    }

    void kill_iter(const std::string &it_var) {
        auto [range_begin, range_end] = iter_dep_constants_.equal_range(it_var);
        for (auto it_const = range_begin; it_const != range_end;) {
            auto &[name, indices] = it_const->second;
            constants_[name].erase(indices);

            auto prev_it_const = it_const++;
            iter_dep_constants_.erase(prev_it_const);
        }
    }

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
        std::unordered_map<std::string, std::map<ScalarIndices, Expr>> other) {
        bool changed = false;
        for (auto &[var, curr_scalar_dict] : constants_) {
            // The outer map is maintained according to VarDef, thus should
            // always exist
            ASSERT(other.count(var));
            auto &other_scalar_dict = other[var];
            // Iterate with manually maintained iterator to allow `erase`
            for (auto it = curr_scalar_dict.cbegin();
                 it != curr_scalar_dict.cend();) {
                bool must_delete = true;
                auto &[idx, curr_val] = *it;
                // If the same scalar exists, check for equivalence
                if (other_scalar_dict.count(idx)) {
                    auto &other_val = other_scalar_dict[idx];
                    bool equal = HashComparator()(other_val, curr_val);
                    if (equal)
                        must_delete = false;
                }
                // advance iterator
                it++;
                // do delete
                if (must_delete) {
                    kill_constant(var, idx);
                    changed = true;
                }
            }
        }
        return changed;
    }

  protected:
    /// Store: kill & gen optionally
    Stmt visit(const Store &store_orig) override {
        auto store_unchecked = Mutator::visit(store_orig);
        ASSERT(store_unchecked->nodeType() == ASTNodeType::Store);
        auto store = store_unchecked.as<StoreNode>();

        // const map is maintained according to VarDefs, should always find
        ASSERT(constants_.count(store->var_));

        // convert constant value type first
        auto expr = store->expr_;
        if (expr->isConst())
            expr = castType(tensors_type_[store->var_],
                            store->expr_.as<ConstNode>());

        // generate constant value
        gen_constant(store->var_, tryToScalar(store->indices_), expr);

        return store;
    }

    /// ReduceTo: kill & gen optionally
    Stmt visit(const ReduceTo &op) override {
        return makeReduction(visitStmt(undoMakeReduction(op)));
    }

    /// Load: read from constants map
    Expr visit(const Load &load_orig) override {
        auto load_unchecked = Mutator::visit(load_orig);
        ASSERT(load_unchecked->nodeType() == ASTNodeType::Load);
        auto load = load_unchecked.as<LoadNode>();

        // if is scalar and found in constants map, return it
        auto indices = tryToScalar(load->indices_);
        if (indices && constants_[load->var_].count(*indices))
            return deepCopy(constants_[load->var_][*indices]);

        return load;
    }

    /// If: choose single branch if constant predicate, or intersect both
    /// branches
    Stmt visit(const If &op) override {
        auto cond = visitExpr(op->cond_);
        if (cond->nodeType() == ASTNodeType::BoolConst) {
            // constant branch, eliminate one
            if (cond.as<BoolConstNode>()->val_)
                return visitStmt(op->thenCase_);
            else
                return op->elseCase_.isValid() ? visitStmt(op->elseCase_)
                                               : makeStmtSeq("", {});
        } else {
            // keep both branches, propagate on each one

            // backup current map for else branch
            std::pair backup = {constants_, iter_dep_constants_};
            auto then_case = visitStmt(op->thenCase_);
            // record then branch result and recover previous for else branch
            auto then_constants = constants_;
            constants_ = std::move(backup.first);
            iter_dep_constants_ = std::move(backup.second);
            // walk else branch
            auto else_case =
                op->elseCase_.isValid() ? visitStmt(op->elseCase_) : nullptr;
            // intersect both branches
            intersect_constants_with(std::move(then_constants));
            // reconstruct If node
            return makeIf(op->id(), cond, then_case, else_case);
        }
    }

    /// VarDef: maintain top level of the maps
    Stmt visit(const VarDef &vd) override {
        auto &name = vd->name_;
        auto dtype = vd->buffer_->tensor().dtype();
        // create entry for constant and type map
        constants_[name] = std::map<ScalarIndices, Expr>();
        tensors_type_[name] = dtype;
        // generic visit
        auto res_vd = Mutator::visit(vd);
        // remove self entry
        kill_constant(name, std::nullopt);
        constants_.erase(name);
        tensors_type_.erase(name);
        return res_vd;
    }

    /// For: iterate until fixed-point reached
    Stmt visit(const For &op) override {
        // Since we aren't aware of loop times in this scalar pass, we treat it
        // as any iterations, thus a fixed-point is required
        int iter_times = 0;
        while (true) {
            // backup constants before iteration
            std::pair backup = {constants_, iter_dep_constants_};
            // generic visit for one iteration
            Mutator::visit(op);

            kill_iter(op->iter_);

            // intersect with pre-loop map and seek for a fixed-point:

            // swap backup and current to ensure intersect returns changed
            // across iteration
            std::swap(backup.first, constants_);
            std::swap(backup.second, iter_dep_constants_);
            // do intersect
            if (!intersect_constants_with(backup.first))
                break;

            // check dangerously many iterations
            if (iter_times++ == 100)
                WARNING("ScalarPropConst on For loop iterated over 100 times");
        }
        // propagate on body again to get post- fixed-point code
        return Mutator::visit(op);
    }

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
    // Macros for operators.
    // SFINAE tricks are heavily used for reporting error on invalid types.
#define BINARY_OP(OPNAME, OP)                                                  \
    struct op_f_##OPNAME {                                                     \
        template <typename T, typename U>                                      \
        auto operator()(const T &l, const U &r, int) -> decltype(l OP r) {     \
            return l OP r;                                                     \
        }                                                                      \
        template <typename T, typename U>                                      \
        auto operator()(const T &l, const U &r, char) -> decltype(l) {         \
            ERROR("Invalid operator " #OPNAME " on given types");              \
            return l;                                                          \
        }                                                                      \
    };                                                                         \
    Expr visit(const OPNAME &op) override {                                    \
        return visitBinary(                                                    \
            op, [](auto l, auto r) { return op_f_##OPNAME()(l, r, 0); },       \
            [](auto l, auto r) { return make##OPNAME(l, r); });                \
    }
#define BINARY_OP_F(OPNAME, OP_F, OP_TYPE_HINT)                                \
    struct op_f_##OPNAME {                                                     \
        template <typename T, typename U>                                      \
        auto operator()(const T &l, const U &r, int)                           \
            -> decltype(l OP_TYPE_HINT r) {                                    \
            typedef decltype(l OP_TYPE_HINT r) V;                              \
            return (OP_F)((V)l, (V)r);                                         \
        }                                                                      \
        template <typename T, typename U>                                      \
        auto operator()(const T &l, const U &r, char) -> decltype(l) {         \
            ERROR("Invalid operator " #OPNAME " on given types");              \
            return l;                                                          \
        }                                                                      \
    };                                                                         \
    Expr visit(const OPNAME &op) override {                                    \
        return visitBinary(                                                    \
            op, [](auto l, auto r) { return op_f_##OPNAME()(l, r, 0); },       \
            [](auto l, auto r) { return make##OPNAME(l, r); });                \
    }
#define UNARY_OP(OPNAME, OP)                                                   \
    struct op_f_##OPNAME {                                                     \
        template <typename T>                                                  \
        auto operator()(const T &t, int) -> decltype(OP(t)) {                  \
            return OP(t);                                                      \
        }                                                                      \
        template <typename T>                                                  \
        auto operator()(const T &t, char) -> decltype(t) {                     \
            ERROR("Invalid operator " #OPNAME " on given types");              \
            return t;                                                          \
        }                                                                      \
    };                                                                         \
    Expr visit(const OPNAME &op) override {                                    \
        return visitUnary(                                                     \
            op, [](auto x) { return op_f_##OPNAME()(x, 0); },                  \
            [](auto x) { return make##OPNAME(x); });                           \
    }

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

    Expr visit(const Cast &op) {
        auto expr = visitExpr(op->expr_);
        if (expr->isConst() &&
            (op->dtype_ == DataType::Bool || op->dtype_ == DataType::Float32 ||
             op->dtype_ == DataType::Float64 ||
             op->dtype_ == DataType::Int32)) {
            expr = castType(op->dtype_, expr.as<ConstNode>());
        }
        return COPY_DEBUG_INFO(makeCast(expr, op->dtype_), op);
    }
}; // namespace ir

Stmt scalarPropConst(const Stmt &op) { return ScalarPropConst()(op); }

} // namespace ir