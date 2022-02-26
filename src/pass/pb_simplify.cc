#include <mangle.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/pb_simplify.h>

#include <itertools.hpp>

namespace ir {

template <class T>
static void appendTo(std::vector<T> &target, const std::vector<T> &other) {
    target.insert(target.end(), other.begin(), other.end());
}

void PBCompBounds::visitExpr(const Expr &op) {
    CompUniqueBounds::visitExpr(op);

    if (visited_.count(op)) {
        return;
    }
    visited_.insert(op);

    if (!isInt(dtype(op))) {
        return;
    }
    if (auto &&expr = genPBExpr_.gen(op); expr.isValid()) {
        auto &&vars = genPBExpr_.vars(op);

        // We use the original conditions instead of relying on transient bounds
        // here. E.g., for x + y <= 2, and we are computing the maximum value of
        // x + y, we shall not rely on x < 2 - y and y < 2 - x. Instead, we use
        // x + y < 2 directly
        std::vector<std::string> condExprs;
        for (auto &&cond : transients_.conds()) {
            if (auto &&condExpr = genPBExpr_.gen(cond); condExpr.isValid()) {
                for (auto &&var : genPBExpr_.vars(cond)) {
                    if (!vars.count(var.first)) {
                        goto ignore;
                    }
                }
                condExprs.emplace_back(*condExpr);
            ignore:;
            }
        }

        std::string str = "{[";
        for (auto &&[i, var] : iter::enumerate(vars)) {
            str += (i == 0 ? "" : ", ") + var.second;
        }
        str += "] -> [" + *expr + "]";
        for (auto &&[i, cond] : iter::enumerate(condExprs)) {
            str += (i == 0 ? ": " : " and ") + cond;
        }
        str += "}";
        PBMap map(isl_, str);
        PBSet image = range(std::move(map));
        PBVal maxVal = dimMaxVal(image, 0);
        if (maxVal.isRat()) {
            auto &&list = getUpper(op);
            auto maxP = maxVal.numSi();
            auto maxQ = maxVal.denSi();
            updUpper(list, UpperBound{LinearExpr<Rational<int64_t>>{
                               {}, Rational<int64_t>{maxP, maxQ}}});
            setUpper(op, std::move(list));
        }
        PBVal minVal = dimMinVal(image, 0);
        if (minVal.isRat()) {
            auto &&list = getLower(op);
            auto minP = minVal.numSi();
            auto minQ = minVal.denSi();
            updLower(list, LowerBound{LinearExpr<Rational<int64_t>>{
                               {}, Rational<int64_t>{minP, minQ}}});
            setLower(op, std::move(list));
        }
    }
}

Stmt pbSimplify(const Stmt &op) {
    return flattenStmtSeq(std::get<0>(simplifyAndGetBounds<PBSimplify>(op)));
}

} // namespace ir
