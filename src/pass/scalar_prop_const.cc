
#include <pass/scalar_prop_const.h>
#include <pass/undo_make_reduction.h>

#include <mutator.h>

#include <map>
#include <stack>

namespace ir {

/**
 * Mutator for propagating scalar constants.
 * Scalars are values in tensors indexed with constants, i.e. this pass
 * requires both indices and assigned value to be constants.
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
    std::unordered_map<std::string, std::map<ScalarIndices, Const>> constants_;
    /// Type of currently available `vardef`s
    std::unordered_map<std::string, DataType> tensors_type_;

    /**
     * @brief Cast the data type of a `Const` node.
     *
     * @param type Target type
     * @param val Constant node to be casted
     * @return Const Casted constant node
     */
    static Const castType(DataType type, const Const &val) {
        return dispatch(val, [type](auto v) {
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
        std::unordered_map<std::string, std::map<ScalarIndices, Const>> other) {
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
                    // Constant map should always store as target type
                    ASSERT(other_val->nodeType() == curr_val->nodeType());
                    // manually capture curr_val to workaround structural
                    // binding
                    bool equal = dispatch(
                        other_val, [curr_val = curr_val](auto other_data) {
                            return dispatch(curr_val, [&](auto curr_data) {
                                return other_data == curr_data;
                            });
                        });
                    if (equal)
                        must_delete = false;
                }
                // advance and keep previous iterator
                auto prev_it = it++;
                // do delete
                if (must_delete)
                    curr_scalar_dict.erase(prev_it);
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

        // try converting to scalar indices
        auto indices = tryToScalar(store->indices_);
        if (!indices) {
            // not scalar store, kill entire tensor
            constants_[store->var_].clear();
        } else {
            // scalar store, kill scalar and gen if constant value provided
            if (store->expr_->isConst())
                // cast type to target tensor
                constants_[store->var_][*indices] = castType(
                    tensors_type_[store->var_], store->expr_.as<ConstNode>());
            else
                constants_[store->var_].erase(*indices);
        }

        return store;
    }

    /// ReduceTo: kill & gen optionally
    Stmt visit(const ReduceTo &reduce_orig) override {
        auto reduce_unchecked = Mutator::visit(reduce_orig);
        ASSERT(reduce_unchecked->nodeType() == ASTNodeType::ReduceTo);
        auto reduce = reduce_unchecked.as<ReduceToNode>();

        // const map is maintained according to VarDefs, should always find
        ASSERT(constants_.count(reduce->var_));

        // try converting to scalar indices
        auto indices = tryToScalar(reduce->indices_);
        if (!indices) {
            // not scalar reduction, kill entire tensor
            constants_[reduce->var_].clear();
        } else {
            // scalar reduction, kill scalar and gen if constant value produced

            // ReduceTo only produces constant result when both sides has
            // constant value
            if (constants_[reduce->var_].count(*indices) &&
                reduce->expr_->isConst()) {
                Expr result;
                // compute reduction by creating a node and fold it
                switch (reduce->op_) {
                case ReduceOp::Add:
                    result = makeAdd(constants_[reduce->var_][*indices],
                                     reduce->expr_);
                    break;
                case ReduceOp::Mul:
                    result = makeMul(constants_[reduce->var_][*indices],
                                     reduce->expr_);
                    break;
                case ReduceOp::Min:
                    result = makeMin(constants_[reduce->var_][*indices],
                                     reduce->expr_);
                    break;
                case ReduceOp::Max:
                    result = makeMax(constants_[reduce->var_][*indices],
                                     reduce->expr_);
                    break;
                default:
                    ASSERT(false);
                }
                result = visitExpr(result);
                // cast type to target tensor
                constants_[reduce->var_][*indices] = castType(
                    tensors_type_[reduce->var_], result.as<ConstNode>());
            } else
                constants_[reduce->var_].erase(*indices);
        }

        return reduce;
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
            auto backup_constants = constants_;
            auto then_case = visitStmt(op->thenCase_);
            // record then branch result and recover previous for else branch
            auto then_constants = constants_;
            constants_ = std::move(backup_constants);
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
        constants_[name] = std::map<ScalarIndices, Const>();
        tensors_type_[name] = dtype;
        // generic visit
        auto res_vd = Mutator::visit(vd);
        // remove self entry
        constants_.erase(name);
        tensors_type_.erase(name);
        return res_vd;
    }

    /// For: iterate until fixed-point reached
    Stmt visit(const For &op) override {
        // Since we aren't aware of loop times in this scalar pass, we treat it
        // as any iterations, thus a fixed-point is required
        while (true) {
            // backup constants before iteration
            auto before_loop = constants_;
            // generic visit for one iteration
            auto res = Mutator::visit(op);
            // intersect with pre-loop map and seek for a fixed-point
            if (!intersect_constants_with(before_loop))
                return res;
        }
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
#define BINARY_OP(OPNAME, OP)                                                  \
    Expr visit(const OPNAME &op) override {                                    \
        return visitBinary(                                                    \
            op, [](auto l, auto r) { return l OP r; },                         \
            [](auto l, auto r) { return make##OPNAME(l, r); });                \
    }
#define UNARY_OP(OPNAME, OP)                                                   \
    Expr visit(const OPNAME &op) override {                                    \
        return visitUnary(                                                     \
            op, [](auto x) { return OP x; },                                   \
            [](auto x) { return make##OPNAME(x); });                           \
    }

    BINARY_OP(Add, +)
    BINARY_OP(Sub, -)
    BINARY_OP(LAnd, &&)
    BINARY_OP(LOr, ||)
    BINARY_OP(EQ, ==)
    BINARY_OP(NE, !=)
    BINARY_OP(LT, <)
    BINARY_OP(LE, <=)
    BINARY_OP(GT, >)
    BINARY_OP(GE, >=)
    UNARY_OP(LNot, !)

    Expr visit(const Min &op) override {
        return visitBinary(
            op,
            [](auto l, auto r) {
                typedef decltype(l + r) T;
                return std::min((T)l, (T)r);
            },
            [](auto l, auto r) { return makeMin(l, r); });
    }

    Expr visit(const Max &op) override {
        return visitBinary(
            op,
            [](auto l, auto r) {
                typedef decltype(l + r) T;
                return std::max((T)l, (T)r);
            },
            [](auto l, auto r) { return makeMax(l, r); });
    }
};

Stmt scalarPropConst(const Stmt &op) { return ScalarPropConst()(op); }

} // namespace ir