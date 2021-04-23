#include <functional>
#include <unordered_map>

#include <analyze/hash.h>
#include <math/min_max.h>

namespace ir {

namespace {

typedef std::function<Expr(const Expr &, const Expr &)> MakerType;

auto makeMinFunc = [](const Expr &lhs, const Expr &rhs) {
    return makeMin(lhs, rhs);
};
auto makeMaxFunc = [](const Expr &lhs, const Expr &rhs) {
    return makeMax(lhs, rhs);
};

Expr makeOuterInner(
    const MakerType &makeOuter, const MakerType &makeInner,
    std::vector<std::unordered_map<uint64_t, Expr>>::iterator begin,
    std::vector<std::unordered_map<uint64_t, Expr>>::iterator end) {
    std::unordered_map<uint64_t, int> counter;
    uint64_t mostHash;
    Expr mostExpr;
    int mostCnt = 0;
    for (auto it = begin; it != end; it++) {
        auto &&group = *it;
        for (auto &&item : group) {
            int cnt = ++counter[item.first];
            if (cnt > mostCnt) {
                mostHash = item.first;
                mostExpr = item.second;
                mostCnt = cnt;
            }
        }
    }
    if (!mostExpr.isValid()) {
        return nullptr;
    }

    auto split = begin;
    for (auto i = begin; i != end; i++) {
        if (i->count(mostHash)) {
            std::swap(*split, *i);
            split++;
        }
    }
    ASSERT(begin != split);
    for (auto it = begin; it != split; it++) {
        it->erase(mostHash);
    }

    auto left = makeOuterInner(makeOuter, makeInner, begin, split);
    left = left.isValid() ? makeInner(mostExpr, left) : mostExpr;
    if (split != end) {
        auto right = makeOuterInner(makeOuter, makeInner, split, end);
        left = right.isValid() ? makeOuter(left, right) : left;
    }
    return left;
}

Expr makeOuterInner(const MakerType &makeOuter, const MakerType &makeInner,
                    const std::vector<std::vector<Expr>> &exprs) {
    std::vector<std::unordered_map<uint64_t, Expr>> exprsMap;
    exprsMap.reserve(exprs.size());
    for (auto &&group : exprs) {
        std::unordered_map<uint64_t, Expr> groupMap;
        for (auto &&item : group) {
            auto h = getHash(item);
            groupMap[h] = item;
        }
        exprsMap.emplace_back(std::move(groupMap));
    }
    auto ret =
        makeOuterInner(makeOuter, makeInner, exprsMap.begin(), exprsMap.end());
    ASSERT(ret.isValid());
    return ret;
}

} // Anonymous namespace

Expr makeMinMax(const std::vector<std::vector<Expr>> &exprs) {
    return makeOuterInner(makeMinFunc, makeMaxFunc, exprs);
}

Expr makeMaxMin(const std::vector<std::vector<Expr>> &exprs) {
    return makeOuterInner(makeMaxFunc, makeMinFunc, exprs);
}

} // namespace ir

