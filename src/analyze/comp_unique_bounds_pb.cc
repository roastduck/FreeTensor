#include <climits>
#include <string>
#include <unordered_map>

#include <analyze/all_uses.h>
#include <analyze/comp_unique_bounds_pb.h>
#include <analyze/normalize_conditional_expr.h>
#include <container_utils.h>
#include <expr.h>
#include <math/parse_pb_expr.h>
#include <math/presburger.h>
#include <math/utils.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/replace_iter.h>
#include <serialize/mangle.h>
#include <type/data_type.h>

namespace freetensor {

int64_t CompUniqueBoundsPB::Bound::lowerInt() const {
    auto lower = dimMinVal(bound_, 0);
    if (lower.isNegInf())
        return LLONG_MIN;
    return floorDiv(lower.numSi(), lower.denSi());
}
int64_t CompUniqueBoundsPB::Bound::upperInt() const {
    auto upper = dimMaxVal(bound_, 0);
    if (upper.isInf())
        return LLONG_MAX;
    return ceilDiv(upper.numSi(), upper.denSi());
}
std::optional<int64_t> CompUniqueBoundsPB::Bound::getInt() const {
    auto upper = dimFixVal(bound_, 0);
    if (!upper.isInt())
        return std::nullopt;
    ASSERT(upper.denSi() == 1);
    return upper.numSi();
}

namespace {

Expr translateBoundFunc(
    const PBSet &boundSet,
    const std::unordered_map<std::string, Expr> &demangleMap) {

    if (boundSet.empty()) {
        return nullptr;
    }

    // TODO: clear out those not related params
    PBSet compactedBoundSet = coalesce(boundSet);
    auto parsed = parsePBFuncReconstructMinMax(compactedBoundSet);

    Expr result;
    ReplaceIter demangler(demangleMap);
    for (auto piece : views::reverse(parsed)) {
        for (auto &&arg : piece.args_)
            ASSERT(demangleMap.contains(arg));
        ASSERT(piece.values_.size() == 1);
        auto pieceExpr = demangler(piece.values_[0]);
        if (piece.cond_.isValid()) {
            auto condExpr = demangler(piece.cond_);
            result = result.isValid() ? makeIfExpr(condExpr, pieceExpr, result)
                                      : pieceExpr;
        } else {
            result = pieceExpr;
        }
    }
    return result;
}

} // namespace

Expr CompUniqueBoundsPB::Bound::lowerExpr() const {
    return bound_.hasLowerBound(0)
               ? translateBoundFunc(lexmin(bound_), *demangleMap_)
               : nullptr;
}
Expr CompUniqueBoundsPB::Bound::upperExpr() const {
    return bound_.hasUpperBound(0)
               ? translateBoundFunc(lexmax(bound_), *demangleMap_)
               : nullptr;
}
std::tuple<Expr, Expr, Expr>
CompUniqueBoundsPB::Bound::lowerUpperDiffExpr() const {
    PBSet l = bound_.hasLowerBound(0) ? lexmin(bound_) : PBSet();
    PBSet u = bound_.hasUpperBound(0) ? lexmax(bound_) : PBSet();
    PBSet diff =
        l.isValid() && u.isValid()
            ? coalesce(apply(cartesianProduct(u, l),
                             PBMap(bound_.ctx(), "{[u, l] -> [u - l]}")))
            : PBSet();
    return {l.isValid() ? translateBoundFunc(l, *demangleMap_) : nullptr,
            u.isValid() ? translateBoundFunc(u, *demangleMap_) : nullptr,
            diff.isValid() ? translateBoundFunc(diff, *demangleMap_) : nullptr};
}

Ref<CompUniqueBounds::Bound> CompUniqueBoundsPB::Bound::restrictScope(
    const std::unordered_set<std::string> &scope) const {
    std::vector<int> axesToProject;
    for (int i = 0; i < bound_.nParamDims(); ++i) {
        for (auto &&used : allNames(demangleMap_->at(bound_.nameParamDim(i)))) {
            if (!scope.contains(used)) {
                axesToProject.emplace_back(i);
                break;
            }
        }
    }
    auto newBound = bound_;
    for (auto axes : views::reverse(axesToProject))
        newBound = projectOutParamDims(newBound, axes, 1);
    return Ref<CompUniqueBoundsPB::Bound>::make(demangleMap_, newBound);
}

Expr CompUniqueBoundsPB::Bound::simplestExpr(
    const Expr &reference,
    const std::unordered_map<std::string, int> &orderedScope) const {

    // first test the original map to be single valued
    if (!bound_.isSingleValued())
        return nullptr;

    std::vector<std::pair<std::string, int>> axesScopeLevel;
    for (int i = 0; i < bound_.nParamDims(); ++i) {
        auto name = bound_.nameParamDim(i);
        axesScopeLevel.emplace_back(
            name, countScope(demangleMap_->at(name), orderedScope));
    }
    // sort to innermost first, we will try remove them one by one
    std::sort(axesScopeLevel.begin(), axesScopeLevel.end(),
              [](auto &&a, auto &&b) { return a.second > b.second; });

    // remove one axis at a time, try until it's not single valued
    auto restrictedBound = bound_;
    int minScopeLevel = INT_MAX,
        oldScopeLevel = countScope(reference, orderedScope);
    for (auto &&[axis, scopeLevel] : axesScopeLevel) {
        auto newRestrictedBound =
            projectOutParamById(std::move(restrictedBound), axis);
        if (!newRestrictedBound.isSingleValued())
            break;
        restrictedBound = std::move(newRestrictedBound);
        minScopeLevel = scopeLevel;
    }
    auto resultExpr = translateBoundFunc(restrictedBound, *demangleMap_);
    if (!resultExpr.isValid()) {
        return nullptr;
    }
    auto isSimplier = minScopeLevel < oldScopeLevel ||
                      countHeavyOps(resultExpr) < countHeavyOps(reference);
    return isSimplier ? resultExpr : nullptr;
}

CompUniqueBoundsPB::CompUniqueBoundsPB(
    const CompTransientBoundsInterface &transients)
    : CompUniqueBounds(transients), transients_(transients),
      ctx_(Ref<PBCtx>::make()) {

    // construct full condition
    Expr fullCond = makeBoolConst(true);
    for (auto &&cond : transients_.conds())
        fullCond = makeLAnd(fullCond, cond);

    // generate PB condition
    std::string str;
    GenPBExpr::VarMap varMap;
    for (auto &&[subExpr, cond] : normalizeConditionalExpr(fullCond)) {
        auto [subStr, subVarMap] = genPBExpr_.gen(subExpr);
        subStr = "[unique_bounded_var] : " + subStr;
        for (auto &&[k, v] : subVarMap) {
            if (auto it = varMap.find(k); it != varMap.end()) {
                ASSERT(it->second == v);
            } else {
                varMap[k] = v;
            }
        }
        if (cond.isValid()) {
            auto [condStr, condVarMap] = genPBExpr_.gen(cond);
            subStr += " and " + condStr;
            for (auto &&[k, v] : condVarMap) {
                if (auto it = varMap.find(k); it != varMap.end()) {
                    ASSERT(it->second == v);
                } else {
                    varMap[k] = v;
                }
            }
        }
        str += str.empty() ? subStr : "; " + subStr;
    }
    cachedConds_ = PBSet(ctx_, "[" + (varMap | views::values | join(", ")) +
                                   "] -> {" + str + "}");

    // initialize known demangle map
    cachedFreeVars_ = decltype(cachedFreeVars_)::make();
    for (auto &&[expr, pbVar] : varMap) {
        ASSERT(!cachedFreeVars_->contains(pbVar));
        (*cachedFreeVars_)[pbVar] = expr;
    }
}

Ref<CompUniqueBounds::Bound> CompUniqueBoundsPB::getBound(const Expr &op) {
    if (!isInt(op->dtype()))
        return nullptr;

    // find in cached results
    if (auto it = cachedValues_.find(op); it != cachedValues_.end())
        return it->second;

    // not previously queried, construct the bound
    std::string str;
    GenPBExpr::VarMap varMap;
    for (auto &&[subExpr, cond] : normalizeConditionalExpr(op)) {
        auto [subStr, subVarMap] = genPBExpr_.gen(subExpr);
        subStr = "[" + subStr + "]";
        for (auto &&[k, v] : subVarMap) {
            if (auto it = varMap.find(k); it != varMap.end()) {
                ASSERT(it->second == v);
            } else {
                varMap[k] = v;
            }
        }
        if (cond.isValid()) {
            auto [condStr, condVarMap] = genPBExpr_.gen(cond);
            subStr += " : " + condStr;
            for (auto &&[k, v] : condVarMap) {
                if (auto it = varMap.find(k); it != varMap.end()) {
                    ASSERT(it->second == v);
                } else {
                    varMap[k] = v;
                }
            }
        }
        str += str.empty() ? subStr : "; " + subStr;
    }
    auto bound =
        (intersect(PBSet(ctx_, "[" + (varMap | views::values | join(", ")) +
                                   "] -> {" + str + "}"),
                   cachedConds_));
    // update free variables
    for (auto &&[expr, pbVar] : varMap) {
        if (auto it = cachedFreeVars_->find(pbVar);
            it != cachedFreeVars_->end())
            ASSERT(HashComparator()(it->second, expr));
        else
            (*cachedFreeVars_)[pbVar] = expr;
    }
    return cachedValues_[op] = Ref<Bound>::make(cachedFreeVars_, bound);
}

bool CompUniqueBoundsPB::alwaysLE(const Expr &lhs, const Expr &rhs) {
    auto l = insertDims(getBound(lhs).as<Bound>()->bound_, 1, 1),
         r = insertDims(getBound(rhs).as<Bound>()->bound_, 0, 1);
    // we check for the emptiness of l > r; if empty, it means we never have l >
    // r, or equivalently always have l <= r
    auto combined = intersect(intersect(l, r), PBSet(ctx_, "{[l, r]: l > r}"));
    return combined.empty();
}

bool CompUniqueBoundsPB::alwaysLT(const Expr &lhs, const Expr &rhs) {
    auto l = insertDims(getBound(lhs).as<Bound>()->bound_, 1, 1),
         r = insertDims(getBound(rhs).as<Bound>()->bound_, 0, 1);
    // similar to alwaysLE, but !LT = GE
    auto combined = intersect(intersect(l, r), PBSet(ctx_, "{[l, r]: l >= r}"));
    return combined.empty();
}

Ref<CompUniqueBoundsPB::Bound> CompUniqueBoundsPB::unionBoundsAsBound(
    const std::vector<Ref<CompUniqueBounds::Bound>> &_bounds) {
    if (_bounds.size() == 0)
        return nullptr;

    // PBSet in _bounds may be from foreign ctx. Reconstruct them in our ctx
    auto bounds = ranges::to<std::vector>(
        _bounds | views::transform([&](auto &&_bound) {
            ASSERT(_bound->type() == BoundType::Presburger);
            auto &&bound = _bound.template as<Bound>();
            return Ref<Bound>::make(bound->demangleMap_,
                                    bound->bound_.to(ctx_));
        }));

    // union the bounds
    PBSet bound = bounds[0]->bound_;
    for (size_t i = 1; i < bounds.size(); ++i) {
        bound = uni(std::move(bound), bounds[i]->bound_);
    }
    bound = coalesce(std::move(bound));

    // construct the demangle map
    auto demangleMap = Ref<std::unordered_map<std::string, Expr>>::make();
    for (isl_size dim = 0; dim < bound.nParamDims(); ++dim) {
        auto dimName = bound.nameParamDim(dim);
        Expr demangled;
        for (const auto &srcBound : bounds) {
            auto &&srcDemangleMap = *srcBound.as<Bound>()->demangleMap_;
            auto it = srcDemangleMap.find(dimName);
            if (it != srcDemangleMap.end()) {
                if (demangled.isValid()) {
                    ASSERT(HashComparator{}(demangled, it->second));
                } else {
                    demangled = it->second;
                }
            }
        }
        (*demangleMap)[dimName] = demangled;
    }

    return Ref<CompUniqueBoundsPB::Bound>::make(demangleMap, bound);
}

std::pair<Expr, Expr> CompUniqueBoundsPB::unionBounds(
    const std::vector<Ref<CompUniqueBounds::Bound>> &bounds) {
    auto bound = unionBoundsAsBound(bounds);

    // if no bound presented, return an empty range
    if (!bound.isValid()) {
        return {makeIntConst(0), makeIntConst(-1)};
    }

    // translate the lower and upper bounds back to expression
    return {bound->lowerExpr(), bound->upperExpr()};
}

} // namespace freetensor
