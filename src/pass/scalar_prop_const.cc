
#include <analyze/all_iters.h>
#include <analyze/all_uses.h>

#include <pass/make_reduction.h>
#include <pass/scalar_prop_const.h>
#include <pass/undo_make_reduction.h>

#include <math/utils.h>

#include <hash.h>

#include <cmath>
#include <map>
#include <stack>

namespace ir {

std::optional<ScalarPropConst::ScalarIndices>
ScalarPropConst::tryToScalar(const std::vector<SubTree<ExprNode>> &exprs) {
    ScalarIndices res;
    for (auto &i : exprs)
        if (i->nodeType() == ASTNodeType::IntConst)
            res.offset.push_back(i.as<IntConstNode>()->val_);
        else
            return std::nullopt;
    return std::move(res);
}

void ScalarPropConst::gen_constant(const std::string &name,
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

void ScalarPropConst::kill_iter_dep_entry(const std::string &name,
                                          const ScalarIndices &indices) {
    for (auto &it_var : allIters(constants_[name][indices])) {
        auto [range_begin, range_end] = iter_dep_constants_.equal_range(it_var);
        for (auto it_const = range_begin; it_const != range_end;) {
            auto prev_it_const = it_const++;
            if (prev_it_const->second.first == name &&
                prev_it_const->second.second == indices) {
                iter_dep_constants_.erase(prev_it_const);
            }
        }
    }
}

void ScalarPropConst::kill_constant(
    const std::string &name, const std::optional<ScalarIndices> &indices) {
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

void ScalarPropConst::kill_iter(const std::string &it_var) {
    auto [range_begin, range_end] = iter_dep_constants_.equal_range(it_var);
    for (auto it_const = range_begin; it_const != range_end;) {
        auto &[name, indices] = it_const->second;
        constants_[name].erase(indices);

        auto prev_it_const = it_const++;
        iter_dep_constants_.erase(prev_it_const);
    }
}

bool ScalarPropConst::intersect_constants_with(
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
            // do delete
            if (must_delete) {
                // kill_constant(var, idx);
                kill_iter_dep_entry(var, idx);
                it = curr_scalar_dict.erase(it);
                changed = true;
            } else {
                // advance iterator
                it++;
            }
        }
    }
    return changed;
}

/// Store: kill & gen optionally
Stmt ScalarPropConst::visit(const Store &store_orig) {
    auto store_unchecked = BaseClass::visit(store_orig);
    ASSERT(store_unchecked->nodeType() == ASTNodeType::Store);
    auto store = store_unchecked.as<StoreNode>();

    // const map is maintained according to VarDefs, should always find
    ASSERT(constants_.count(store->var_));

    // convert constant value type first
    auto expr = store->expr_;
    if (expr->isConst())
        expr = castType(this->def(store->var_)->buffer_->tensor().dtype(),
                        store->expr_.as<ConstNode>());

    // generate constant value
    gen_constant(store->var_, tryToScalar(store->indices_), expr);

    return store;
}

/// ReduceTo: kill & gen optionally
Stmt ScalarPropConst::visit(const ReduceTo &op) {
    return makeReduction(visitStmt(undoMakeReduction(op)));
}

/// Load: read from constants map
Expr ScalarPropConst::visit(const Load &load_orig) {
    auto load_unchecked = BaseClass::visit(load_orig);
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
Stmt ScalarPropConst::visit(const If &op) {
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
Stmt ScalarPropConst::visit(const VarDef &vd) {
    auto &name = vd->name_;
    // create entry for constant and type map
    constants_[name] = std::map<ScalarIndices, Expr>();
    // generic visit
    auto res_vd = BaseClass::visit(vd);
    // remove self entry
    kill_constant(name, std::nullopt);
    constants_.erase(name);
    return res_vd;
}

/// For: iterate until fixed-point reached
Stmt ScalarPropConst::visit(const For &op) {
    // Since we aren't aware of loop times in this scalar pass, we treat it
    // as any iterations, thus a fixed-point is required
    int iter_times = 0;
    Stmt result;
    while (true) {
        // backup constants before iteration
        std::pair backup = {constants_, iter_dep_constants_};
        // generic visit for one iteration
        result = BaseClass::visit(op);

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
    return result;
}

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
    Expr ScalarPropConst::visit(const OPNAME &op) {                            \
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
    Expr ScalarPropConst::visit(const OPNAME &op) {                            \
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
    Expr ScalarPropConst::visit(const OPNAME &op) {                            \
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

static int64_t _square(const int64_t &t) { return t * t; }
static double _square(const double &t) { return t * t; }

UNARY_OP(Square, _square)
//! TODO: Sigmoid
//! TODO: Tanh

// Avoid -Wabsolute-value in Clang
static int64_t _abs(const int64_t &t) { return std::abs(t); }
static double _abs(const double &t) { return std::abs(t); }

UNARY_OP(Abs, _abs)
UNARY_OP(Floor, std::floor)
UNARY_OP(Ceil, std::ceil)

Expr ScalarPropConst::visit(const Cast &op) {
    auto expr = visitExpr(op->expr_);
    if (expr->isConst() &&
        (op->dtype_ == DataType::Bool || op->dtype_ == DataType::Float32 ||
         op->dtype_ == DataType::Float64 || op->dtype_ == DataType::Int32)) {
        expr = castType(op->dtype_, expr.as<ConstNode>());
    }
    return COPY_DEBUG_INFO(makeCast(expr, op->dtype_), op);
}

Stmt scalarPropConst(const Stmt &op) { return ScalarPropConst()(op); }

} // namespace ir