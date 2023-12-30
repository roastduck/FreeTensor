#include <climits>
#include <string>
#include <unordered_map>

#include <analyze/all_uses.h>
#include <container_utils.h>
#include <expr.h>
#include <math/parse_pb_expr.h>
#include <math/presburger.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/pb_simplify.h>
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

// Translate a computed bound function in ISL back to our Expr.
// We prefer min/max expressions as final results, so we first test if simply
// min/max of the pieces gives the correct result; if not, fallback to IfExpr.
Expr translateBoundFunc(
    PBCtx &ctx, const PBSet &boundSet,
    const std::unordered_map<std::string, Expr> &demangleMap,
    int64_t defaultVal) {

    if (boundSet.empty()) {
        return makeIntConst(defaultVal);
    }

    // TODO: clear out those not related params
    PBSet compactedBoundSet = coalesce(boundSet);
    auto parsed = parsePBFuncReconstructMinMax(ctx, compactedBoundSet);

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
    return translateBoundFunc(*ctx_, lexmin(bound_), *demangleMap_, LLONG_MIN);
}
Expr CompUniqueBoundsPB::Bound::upperExpr() const {
    return translateBoundFunc(*ctx_, lexmax(bound_), *demangleMap_, LLONG_MAX);
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
    return Ref<CompUniqueBoundsPB::Bound>::make(ctx_, demangleMap_, newBound);
}

Expr CompUniqueBoundsPB::Bound::simplestExpr(
    const std::unordered_map<std::string, int> &orderedScope) const {

    // first test the original map to be single valued
    if (!bound_.isSingleValued())
        return nullptr;

    std::vector<std::pair<std::string, int>> axesScopeLevel;
    for (int i = 0; i < bound_.nParamDims(); ++i) {
        auto name = bound_.nameParamDim(i);
        int scopeLevel = 0;
        for (auto &&used : allUses(demangleMap_->at(name)))
            scopeLevel = std::max(scopeLevel, orderedScope.at(used));
        axesScopeLevel.emplace_back(name, scopeLevel);
    }
    // sort to innermost first, we will try remove them one by one
    std::sort(axesScopeLevel.begin(), axesScopeLevel.end(),
              [](auto &&a, auto &&b) { return a.second > b.second; });

    // remove one axis at a time, try until it's not single valued
    auto restrictedBound = bound_;
    for (auto &&[axis, _] : axesScopeLevel) {
        auto newRestrictedBound =
            projectOutParamById(std::move(restrictedBound), axis);
        if (!newRestrictedBound.isSingleValued())
            break;
        restrictedBound = std::move(newRestrictedBound);
    }
    if (!restrictedBound.empty()) {
        return translateBoundFunc(*ctx_, restrictedBound, *demangleMap_,
                                  0 /* any value */);
    } else {
        return nullptr;
    }
}

Ref<CompUniqueBounds::Bound> CompUniqueBoundsPB::getBound(const Expr &op) {
    if (!isInt(op->dtype()))
        return nullptr;

    // check if the cache is valid
    if (auto place = transients_.currentStmt(); place != cachedPlace_) {
        // invalid, refresh it with the new transients condition
        cachedPlace_ = place;

        // construct full condition
        Expr fullCond = makeBoolConst(true);
        for (auto &&cond : transients_.conds())
            fullCond = makeLAnd(fullCond, cond);

        // generate PB condition
        auto [str, varMap] = genPBExpr_.gen(fullCond);
        cachedConds_ =
            PBSet(*ctx_, "[" + (varMap | views::values | join(", ")) +
                             "] -> { [unique_bounded_var]: " + str + " }");

        // initialize known demangle map
        cachedFreeVars_ = decltype(cachedFreeVars_)::make();
        for (auto &&[expr, pbVar] : varMap) {
            ASSERT(!cachedFreeVars_->contains(pbVar));
            (*cachedFreeVars_)[pbVar] = expr;
        }

        // clear cached query results
        cachedValues_.clear();
    }

    // find in cached results
    if (auto it = cachedValues_.find(op); it != cachedValues_.end())
        return it->second;

    // not previously queried, construct the bound
    auto [str, varMap] = genPBExpr_.gen(op);
    auto bound =
        (intersect(PBSet(*ctx_, "[" + (varMap | views::values | join(", ")) +
                                    "] -> { [" + str + "] }"),
                   cachedConds_));
    // update free variables
    for (auto &&[expr, pbVar] : varMap) {
        if (auto it = cachedFreeVars_->find(pbVar);
            it != cachedFreeVars_->end())
            ASSERT(HashComparator()(it->second, expr));
        else
            (*cachedFreeVars_)[pbVar] = expr;
    }
    return cachedValues_[op] = Ref<Bound>::make(ctx_, cachedFreeVars_, bound);
}

bool CompUniqueBoundsPB::alwaysLE(const Expr &lhs, const Expr &rhs) {
    auto l = insertDims(getBound(lhs).as<Bound>()->bound_, 1, 1),
         r = insertDims(getBound(rhs).as<Bound>()->bound_, 0, 1);
    // we check for the emptiness of l > r; if empty, it means we never have l >
    // r, or equivalently always have l <= r
    auto combined = intersect(intersect(l, r), PBSet(*ctx_, "{[l, r]: l > r}"));
    return combined.empty();
}

bool CompUniqueBoundsPB::alwaysLT(const Expr &lhs, const Expr &rhs) {
    auto l = insertDims(getBound(lhs).as<Bound>()->bound_, 1, 1),
         r = insertDims(getBound(rhs).as<Bound>()->bound_, 0, 1);
    // similar to alwaysLE, but !LT = GE
    auto combined =
        intersect(intersect(l, r), PBSet(*ctx_, "{[l, r]: l >= r}"));
    return combined.empty();
}

std::pair<Expr, Expr> CompUniqueBoundsPB::unionBounds(
    const std::vector<Ref<CompUniqueBounds::Bound>> &_bounds) {
    // if no bound presented, return an empty range
    if (_bounds.size() == 0)
        return {makeIntConst(0), makeIntConst(-1)};

    // PBSet in _bounds may be from foreign ctx. Reconstruct them in our ctx
    auto bounds = ranges::to<std::vector>(
        _bounds | views::transform([&](auto &&_bound) {
            auto &&bound = _bound.template as<CompUniqueBoundsPB::Bound>();
            return Ref<CompUniqueBoundsPB::Bound>::make(
                ctx_, bound->demangleMap_,
                PBSet(*ctx_, toString(bound->bound_)));
        }));

    // union the bounds
    PBSet bound = bounds[0].as<Bound>()->bound_;
    for (size_t i = 1; i < bounds.size(); ++i) {
        bound = uni(std::move(bound), bounds[i].as<Bound>()->bound_);
    }
    bound = coalesce(std::move(bound));

    // construct the demangle map
    std::unordered_map<std::string, Expr> demangleMap;
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
        demangleMap[dimName] = demangled;
    }

    // translate the lower and upper bounds back to expression
    return {translateBoundFunc(*ctx_, lexmin(bound), demangleMap, LLONG_MIN),
            translateBoundFunc(*ctx_, lexmax(bound), demangleMap, LLONG_MAX)};
}

Stmt pbSimplify(const Stmt &op) {
    return flattenStmtSeq(simplifyImpl<PBSimplify>(op));
}

} // namespace freetensor
