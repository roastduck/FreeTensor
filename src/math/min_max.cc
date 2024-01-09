#include <functional>
#include <unordered_map>

#include <hash.h>
#include <math/min_max.h>

namespace freetensor {

namespace {

typedef std::function<Expr(const Expr &, const Expr &)> MakerType;

auto makeMinFunc = [](const Expr &lhs, const Expr &rhs) {
    if (lhs->nodeType() == ASTNodeType::IntConst &&
        rhs->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(std::min(lhs.as<IntConstNode>()->val_,
                                     rhs.as<IntConstNode>()->val_));
    }
    return makeMin(lhs, rhs);
};
auto makeMaxFunc = [](const Expr &lhs, const Expr &rhs) {
    if (lhs->nodeType() == ASTNodeType::IntConst &&
        rhs->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(std::max(lhs.as<IntConstNode>()->val_,
                                     rhs.as<IntConstNode>()->val_));
    }
    return makeMax(lhs, rhs);
};
auto makeLAndFunc = [](const Expr &lhs, const Expr &rhs) {
    if (lhs->nodeType() == ASTNodeType::BoolConst) {
        return lhs.as<BoolConstNode>()->val_ ? rhs : makeBoolConst(false);
    }
    if (rhs->nodeType() == ASTNodeType::BoolConst) {
        return rhs.as<BoolConstNode>()->val_ ? lhs : makeBoolConst(false);
    }
    return makeLAnd(lhs, rhs);
};
auto makeLOrFunc = [](const Expr &lhs, const Expr &rhs) {
    if (lhs->nodeType() == ASTNodeType::BoolConst) {
        return lhs.as<BoolConstNode>()->val_ ? makeBoolConst(true) : rhs;
    }
    if (rhs->nodeType() == ASTNodeType::BoolConst) {
        return rhs.as<BoolConstNode>()->val_ ? makeBoolConst(true) : lhs;
    }
    return makeLOr(lhs, rhs);
};

Expr makeOuterInner(const MakerType &makeOuter, const MakerType &makeInner,
                    std::vector<ASTHashSet<Expr>>::iterator begin,
                    std::vector<ASTHashSet<Expr>>::iterator end) {
    // This function assumes the outer term to be non-empty, and returns nullptr
    // in case of any inner term is empty

    for (auto it = begin; it != end; it++) {
        if (it->empty()) {
            // In case of min(max(...), ...), this means min(max(empty), ...) =
            // min(-inf, ...) = -inf
            return nullptr;
        }
    }

    ASTHashMap<Expr, int> counter;
    Expr mostExpr;
    int mostCnt = 0;
    for (auto it = begin; it != end; it++) {
        auto &&group = *it;
        for (auto &&item : group) {
            int cnt = ++counter[item];
            if (cnt > mostCnt) {
                mostExpr = item;
                mostCnt = cnt;
            }
        }
    }
    ASSERT(mostExpr.isValid());

    auto split = begin;
    for (auto i = begin; i != end; i++) {
        if (i->count(mostExpr)) {
            std::swap(*split, *i);
            split++;
        }
    }
    ASSERT(begin != split);
    for (auto it = begin; it != split; it++) {
        it->erase(mostExpr);
    }

    auto left = makeOuterInner(makeOuter, makeInner, begin, split);
    // In case of min(max(...), ...), invalid left <==> max(-inf, mostExpr)
    left = left.isValid() ? makeInner(mostExpr, left) : mostExpr;
    if (split != end) {
        auto right = makeOuterInner(makeOuter, makeInner, split, end);
        // In case of min(max(...), ...), invalid right <==> min(left, -inf)
        left = right.isValid() ? makeOuter(left, right) : nullptr;
    }
    return left;
}

Expr makeOuterInner(const MakerType &makeOuter, const MakerType &makeInner,
                    const std::vector<std::vector<Expr>> &exprs,
                    const std::function<Expr()> &innerEmptyGenerator,
                    const std::function<Expr()> &outerEmptyGenerator) {
    if (exprs.empty()) {
        return outerEmptyGenerator();
    }
    std::vector<ASTHashSet<Expr>> exprsSet;
    exprsSet.reserve(exprs.size());
    for (auto &&group : exprs) {
        ASTHashSet<Expr> groupSet;
        for (auto &&item : group) {
            groupSet.insert(item);
        }
        exprsSet.emplace_back(std::move(groupSet));
    }
    auto ret =
        makeOuterInner(makeOuter, makeInner, exprsSet.begin(), exprsSet.end());
    return ret.isValid() ? ret : innerEmptyGenerator();
}

} // Anonymous namespace

Expr makeMinMaxImpl(const std::vector<std::vector<Expr>> &exprs,
                    const std::function<Expr()> &negInf,
                    const std::function<Expr()> &inf) {
    return makeOuterInner(makeMinFunc, makeMaxFunc, exprs, negInf, inf);
}

Expr makeMaxMinImpl(const std::vector<std::vector<Expr>> &exprs,
                    const std::function<Expr()> &negInf,
                    const std::function<Expr()> &inf) {
    return makeOuterInner(makeMaxFunc, makeMinFunc, exprs, inf, negInf);
}

Expr makeLOrLAnd(const std::vector<std::vector<Expr>> &exprs) {
    return makeOuterInner(
        makeLOrFunc, makeLAndFunc, exprs, []() { return makeBoolConst(true); },
        []() { return makeBoolConst(false); });
}

} // namespace freetensor
