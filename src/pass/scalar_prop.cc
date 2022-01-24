
#include <pass/scalar_prop.h>
#include <pass/undo_make_reduction.h>

#include <mutator.h>

#include <map>
#include <stack>

namespace ir {

/**
 * Visitor for propagating scalars.
 * Scalars are values in tensors indexed with constants.
 */
class ScalarProp : public Mutator {
  private:
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

    struct ScalarIndices {
        std::vector<int64_t> offset;

        bool operator<(const ScalarIndices &other) const {
            ASSERT(offset.size() == other.offset.size() &&
                   "Index count should be identical for same tensor");
            for (size_t i = 0; i < offset.size(); ++i)
                if (offset[i] < other.offset[i])
                    return true;
            return false;
        }
    };
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
    std::unordered_map<std::string, std::map<ScalarIndices, Const>> constants_;
    std::unordered_map<std::string, DataType> tensors_type_;

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

    bool intersect_constants_with(
        std::unordered_map<std::string, std::map<ScalarIndices, Const>> other) {
        bool changed = false;
        for (auto &[var, curr_scalar_dict] : constants_) {
            ASSERT(other.count(var));
            auto &other_scalar_dict = other[var];
            for (auto it = curr_scalar_dict.cbegin();
                 it != curr_scalar_dict.cend();) {
                bool must_delete = true;
                auto &[idx, curr_val] = *it;
                if (other_scalar_dict.count(idx)) {
                    auto &other_val = other_scalar_dict[idx];
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
                auto prev_it = it++;
                if (must_delete)
                    curr_scalar_dict.erase(prev_it);
            }
        }
        return changed;
    }

  protected:
    Stmt visit(const Store &store_orig) override {
        auto store_unchecked = Mutator::visit(store_orig);
        ASSERT(store_unchecked->nodeType() == ASTNodeType::Store);
        auto store = store_unchecked.as<StoreNode>();

        if (constants_.count(store->var_)) {
            auto indices = tryToScalar(store->indices_);
            if (!indices) {
                // kill entire tensor
                constants_[store->var_].clear();
            } else {
                // kill scalar and gen if constant value provided
                if (store->expr_->isConst())
                    constants_[store->var_][*indices] =
                        castType(tensors_type_[store->var_],
                                 store->expr_.as<ConstNode>());
                else
                    constants_[store->var_].erase(*indices);
            }
        }
        return store;
    }

    Stmt visit(const ReduceTo &reduce_orig) override {
        auto reduce_unchecked = Mutator::visit(reduce_orig);
        ASSERT(reduce_unchecked->nodeType() == ASTNodeType::ReduceTo);
        auto reduce = reduce_unchecked.as<ReduceToNode>();

        if (constants_.count(reduce->var_)) {
            auto indices = tryToScalar(reduce->indices_);
            if (!indices) {
                // kill entire tensor
                constants_[reduce->var_].clear();
            } else {
                // kill scalar and gen if constant value provided
                if (constants_[reduce->var_].count(*indices) &&
                    reduce->expr_->isConst()) {
                    Expr result;
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
                    constants_[reduce->var_][*indices] = castType(
                        tensors_type_[reduce->var_], result.as<ConstNode>());
                } else
                    constants_[reduce->var_].erase(*indices);
            }
        }
        return reduce;
    }

    Expr visit(const Load &load_orig) override {
        auto load_unchecked = Mutator::visit(load_orig);
        ASSERT(load_unchecked->nodeType() == ASTNodeType::Load);
        auto load = load_unchecked.as<LoadNode>();

        auto indices = tryToScalar(load->indices_);
        if (indices && constants_[load->var_].count(*indices))
            return deepCopy(constants_[load->var_][*indices]);

        return load;
    }

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
            auto backup_constants = constants_;
            auto then_case = visitStmt(op->thenCase_);
            auto then_constants = constants_;
            constants_ = std::move(backup_constants);
            auto else_case =
                op->elseCase_.isValid() ? visitStmt(op->elseCase_) : nullptr;
            intersect_constants_with(std::move(then_constants));
            return makeIf(op->id(), cond, then_case, else_case);
        }
    }

    Stmt visit(const VarDef &vd) override {
        auto &name = vd->name_;
        auto dtype = vd->buffer_->tensor().dtype();
        constants_[name] = std::map<ScalarIndices, Const>();
        tensors_type_[name] = dtype;
        auto res_vd = Mutator::visit(vd);
        constants_.erase(name);
        tensors_type_.erase(name);
        return res_vd;
    }

    Stmt visit(const For &op) override {
        while (true) {
            auto before_loop = constants_;
            auto res = Mutator::visit(op);
            if (!intersect_constants_with(before_loop))
                return res;
        }
    }

  private:
    template <typename F>
    static Expr doBinary(F f, const Const &l, const Const &r) {
        return dispatch(l, [&](auto ll) {
            return dispatch(r, [&](auto rr) { return wrap(f(ll, rr)); });
        });
    }

    template <typename F, typename FAlt>
    Expr visitBinary(const BinaryExpr &op, F f, FAlt falt) {
        auto lhs = visitExpr(op->lhs_);
        auto rhs = visitExpr(op->rhs_);
        Expr res;
        if (lhs->isConst() && rhs->isConst())
            res = doBinary(f, lhs.as<ConstNode>(), rhs.as<ConstNode>());
        else
            res = falt(lhs, rhs);
        return COPY_DEBUG_INFO(res, op);
    }

    template <typename F> static Expr doUnary(F f, const Const &x) {
        return dispatch(x, [&](auto xx) { return wrap(f(xx)); });
    }

    template <typename F, typename FAlt>
    Expr visitUnary(const UnaryExpr &op, F f, FAlt falt) {
        auto x = visitExpr(op->expr_);
        Expr res;
        if (x->isConst())
            res = doUnary(f, x.as<ConstNode>());
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

Stmt scalarProp(const Stmt &op) { return ScalarProp()(op); }

} // namespace ir