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

Expr makeOuterInner(const MakerType &makeOuter, const MakerType &makeInner,
                    std::vector<ASTHashSet<Expr>>::iterator begin,
                    std::vector<ASTHashSet<Expr>>::iterator end) {
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
                    const std::vector<std::vector<Expr>> &exprs) {
    std::vector<ASTHashSet<Expr>> exprsSet;
    exprsSet.reserve(exprs.size());
    for (auto &&group : exprs) {
        ASTHashSet<Expr> groupSet;
        for (auto &&item : group) {
            groupSet.insert(item);
        }
        exprsSet.emplace_back(std::move(groupSet));
    }
    return makeOuterInner(makeOuter, makeInner, exprsSet.begin(),
                          exprsSet.end());
}

} // Anonymous namespace

Expr makeMinMax(const std::vector<std::vector<Expr>> &exprs) {
    return makeOuterInner(makeMinFunc, makeMaxFunc, exprs);
}

Expr makeMaxMin(const std::vector<std::vector<Expr>> &exprs) {
    return makeOuterInner(makeMaxFunc, makeMinFunc, exprs);
}

} // namespace freetensor
