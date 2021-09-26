#include <pass/flatten_stmt_seq.h>
#include <pass/isl_simplify.h>

#include "detail/simplify.h"

namespace ir {

template <class T>
static void unionTo(std::unordered_set<T> &target,
                    const std::unordered_set<T> &other) {
    target.insert(other.begin(), other.end());
}

template <class T>
static void appendTo(std::vector<T> &target, const std::vector<T> &other) {
    target.insert(target.end(), other.begin(), other.end());
}

void GenISLExprSimplify::visitExpr(
    const Expr &op, const std::function<void(const Expr &)> &visitNode) {
    auto oldParent = parent_;
    parent_ = op;
    GenISLExpr::visitExpr(op, visitNode);
    parent_ = oldParent;
    if (parent_.isValid()) {
        unionTo(vars_[parent_], vars_[op]);
        appendTo(cond_[parent_], cond_[op]);
    }
}

void GenISLExprSimplify::visit(const Var &op) {
    auto str = normalizeId(op->name_);
    vars_[op].insert(str);
    results_[op] = str;
}

void GenISLExprSimplify::visit(const Load &op) {
    getHash_(op);
    auto h = getHash_.hash().at(op);
    auto str = normalizeId("load" + std::to_string(h));
    vars_[op].insert(str);
    results_[op] = str;
}

Expr ISLCompBounds::visitExpr(
    const Expr &_op, const std::function<Expr(const Expr &)> &visitNode) {
    auto op = CompUniqueBounds::visitExpr(_op, visitNode);
    if (!isInt(dtype(op))) {
        return op;
    }
    if (auto &&expr = genISLExpr_.gen(op); expr.isValid()) {
        auto tr = transient(op);
        for (auto &&first : tr.lower_) {
            if (auto &&lowerExpr = genISLExpr_.gen(first);
                lowerExpr.isValid()) {
                for (auto &&var : genISLExpr_.vars(first)) {
                    genISLExpr_.vars(op).insert(var);
                }
                genISLExpr_.cond(op).emplace_back(*expr + " >= " + *lowerExpr);
            }
        }
        for (auto &&second : tr.upper_) {
            if (auto &&upperExpr = genISLExpr_.gen(second);
                upperExpr.isValid()) {
                for (auto &&var : genISLExpr_.vars(second)) {
                    genISLExpr_.vars(op).insert(var);
                }
                genISLExpr_.cond(op).emplace_back(*expr + " <= " + *upperExpr);
            }
        }

        std::string str = "{[";
        bool first = true;
        for (auto &&var : genISLExpr_.vars(op)) {
            str += (first ? "" : ", ") + var;
            first = false;
        }
        str += "] -> [" + *expr + "]";
        first = true;
        for (auto &&cond : genISLExpr_.cond(op)) {
            str += (first ? ": " : " and ") + cond;
            first = false;
        }
        str += "}";
        ISLMap map(isl_, str);
        ISLSet image = range(std::move(map));
        ISLVal maxVal = dimMaxVal(image, 0);
        if (maxVal.isRat()) {
            auto &&list = getUpper(op);
            auto maxP = maxVal.numSi();
            auto maxQ = maxVal.denSi();
            updUpper(list, UpperBound{LinearExpr<Rational<int64_t>>{
                               {}, Rational<int64_t>{maxP, maxQ}}});
            setUpper(op, std::move(list));
        }
        ISLVal minVal = dimMinVal(image, 0);
        if (minVal.isRat()) {
            auto &&list = getLower(op);
            auto minP = minVal.numSi();
            auto minQ = minVal.denSi();
            updLower(list, LowerBound{LinearExpr<Rational<int64_t>>{
                               {}, Rational<int64_t>{minP, minQ}}});
            setLower(op, std::move(list));
        }
    }
    return op;
}

Stmt islSimplify(const Stmt &op) {
    return flattenStmtSeq(std::get<0>(simplifyAndGetBounds<ISLSimplify>(op)));
}

} // namespace ir

