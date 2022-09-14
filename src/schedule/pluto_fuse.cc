#include <cmath>
#include <numeric>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <pass/flatten_stmt_seq.h>
#include <schedule/fuse.h>
#include <schedule/pluto_fuse.h>

namespace freetensor {

namespace {

std::vector<std::string> absSumConstraints(int n, const std::string &sum,
                                           const auto &fVar) {
    return views::ints(0, 1 << n) | views::transform([&](const auto &num) {
               std::ostringstream oss;
               oss << sum << " >= 0";
               for (int i = 0; i < n; ++i) {
                   if ((num >> i) & 1)
                       oss << " + ";
                   else
                       oss << " - ";
                   oss << fVar(i);
               }
               return oss.str();
           }) |
           ranges::to<std::vector>();
}

std::vector<std::string> nonZeroConstraints(int n, const auto &fVar,
                                            const std::string delta,
                                            int varBound = 2) {
    auto rangeConstraints =
        views::ints(0, n) | views::transform([&](int i) {
            return std::array{fVar(i) + " >= -" + toString(varBound),
                              fVar(i) + " <= " + toString(varBound)};
        }) |
        views::join | ranges::to<std::vector>();
    auto skewedExpr =
        views::ints(0, n) | views::transform([&](int i) {
            return toString((int)std::pow(varBound * 2 + 1, i)) + "*" + fVar(i);
        }) |
        join(" + ");
    auto bound = (int)std::pow(varBound * 2 + 1, n);
    auto excludeZero = std::array{
        skewedExpr + " >= 1 - " + toString(bound) + delta,
        skewedExpr + " <= " + toString(bound - 1) + " - " + toString(bound) +
            delta,
        delta + " >= 0",
        delta + " <= 1",
    };
    return views::concat(rangeConstraints, excludeZero) |
           ranges::to<std::vector>();
}

std::vector<std::vector<int>>
orthogonalMatrix(const std::vector<std::vector<int>> &vectors) {
    // sanity check
    ASSERT(vectors.size() > 0);
    for (size_t i = 1; i < vectors.size(); ++i)
        ASSERT(vectors[i].size() == vectors[0].size());
    int nDims = vectors[0].size();

    auto fVar = [](int i) { return "i" + toString(i); };
    auto fOrtho = [&](const std::vector<int> &v) {
        return (views::ints(0, nDims) | views::transform([&](int i) {
                    return toString(v[i]) + fVar(i);
                }) |
                join(" + ")) +
               " = 0";
    };
    auto setHeader =
        "{ [sum, " +
        (views::ints(0, nDims) | views::transform(fVar) | join(", ")) +
        ", delta]: ";
    auto setFooter = " }";

    PBCtx ctx;
    PBSet orthogonalSet;
    {
        auto nonZero = nonZeroConstraints(nDims, fVar, "delta");
        auto absSum = absSumConstraints(nDims, "sum", fVar);
        orthogonalSet = PBSet(
            ctx, setHeader +
                     (views::concat(
                          // orthogonal constraints
                          vectors | views::transform(fOrtho),
                          // non-zero constraints to exclude trivial result
                          nonZero,
                          // abs sum constraints to find vector nearest to zero
                          absSum) |
                      join(" and ")) +
                     setFooter);
    }

    std::vector<std::vector<int>> result;
    while (!orthogonalSet.empty()) {
        // lexmin and sample to solve
        auto solution = sample(lexmin(orthogonalSet)).coordinates();
        // extract the vector into result
        std::vector<int> v;
        v.reserve(solution.size() - 2);
        for (size_t i = 1; i < solution.size() - 1; ++i) {
            ASSERT(solution[i].denSi() == 1);
            v.emplace_back(solution[i].numSi());
        }
        result.emplace_back(std::move(v));
        // inject new constraint to find next orthogonal vector
        PBSet newConstraint(ctx, setHeader + fOrtho(result.back()) + setFooter);
        orthogonalSet = intersect(std::move(orthogonalSet), newConstraint);
    }

    std::cout << ">> Orthogonal Matrix of [ ";
    for (const auto &v : vectors)
        std::cout << "[" << v << "] ";
    std::cout << "] is: [ ";
    for (const auto &v : result)
        std::cout << "[" << v << "] ";
    std::cout << "]" << std::endl;

    return result;
}

constexpr const char *FAKE_ACCESS_VAR = "__pluto_fake_access";

class InjectFakeAccess : public Mutator {
    ID target_;

  public:
    InjectFakeAccess(const ID &target) : target_(target) {}

  protected:
    Stmt visit(const For &op) override {
        if (op->id() != target_)
            return Mutator::visit(op);

        auto newBody = makeStmtSeq({
            makeVarDef(FAKE_ACCESS_VAR,
                       makeBuffer(makeTensor({}, DataType::Int32),
                                  AccessType::Cache, MemType::ByValue),
                       nullptr,
                       makeEval(makeLoad(FAKE_ACCESS_VAR, {}, DataType::Int32)),
                       false),
            op->body_,
        });

        return makeFor(op->iter_, op->begin_, op->end_, op->step_, op->len_,
                       op->property_, newBody);
    }
};

} // namespace

std::pair<Stmt, ID> plutoFuse(const Stmt &_ast, const ID &loop0Id,
                              const ID &loop1Id, int nestLevel) {
    // flatten first so we get perfectly nested loops as much as possible
    auto ast = flattenStmtSeq(_ast);

    // check accessed vardefs: those vardefs accessed by loop1 should not have
    // their shapes modified in loop0
    CheckFuseAccessible check(loop0Id, loop1Id);
    check.check(ast);

    // count maximum count of perfectly nested loops at loop0 and loop1
    auto countPerfectNest = [](const For &loop) {
        int n = 0;
        Stmt inner;
        for (inner = loop; inner->nodeType() == ASTNodeType::For;
             inner = inner.as<ForNode>()->body_)
            n++;
        return std::pair{n, inner->parentStmt()->id()};
    };
    auto [nestLevel0, inner0] = countPerfectNest(check.loop0().loop_);
    auto [nestLevel1, inner1] = countPerfectNest(check.loop1().loop_);

    // inject fake accesses to extract loop space
    ast = InjectFakeAccess(inner0)(ast);
    ast = InjectFakeAccess(inner1)(ast);

    auto loop0 = findStmt(ast, loop0Id).as<ForNode>();
    auto loop1 = findStmt(ast, loop1Id).as<ForNode>();

    // Process nestLevel nested loops each side; default to
    if (nestLevel == -1)
        nestLevel = std::min(nestLevel0, nestLevel1);
    else if (nestLevel0 < nestLevel)
        throw InvalidSchedule(
            "PlutoFuse: loop 0 `#" + toString(loop0Id) +
            "` has less than required nesting levels: " + toString(nestLevel0) +
            " existed, but " + toString(nestLevel) + " required");
    else if (nestLevel1 < nestLevel)
        throw InvalidSchedule(
            "PlutoFuse: loop 1 `#" + toString(loop1Id) +
            "` has less than required nesting levels: " + toString(nestLevel1) +
            " existed, but " + toString(nestLevel) + " required");

    ASSERT(nestLevel > 0);

    // List common outer loops
    std::deque<For> outers;
    for (Stmt outer = loop0->parentStmt(); outer.isValid();
         outer = outer->parentStmt())
        if (outer->nodeType() == ASTNodeType::For)
            outers.push_front(outer.as<ForNode>());

    // Sanity check: the two loops should have the same outer loops given the
    // CheckFuseAccessible passed; just check if the innermost one aligns
    auto loop1InnermostOuter = loop1->parentStmtByFilter(
        [](const Stmt &s) { return s->nodeType() == ASTNodeType::For; });
    // outers are empty, loop1 should also have no outer
    if (outers.empty())
        ASSERT(!loop1InnermostOuter.isValid());
    // outers exist, the innermost outer of loop1 should be the same as loop0
    else
        ASSERT(loop1InnermostOuter == outers.back());

    std::vector<FindDepsDir> outersSame{
        outers | views::transform([&](const For &f) {
            return std::pair{NodeIDOrParallelScope(f->id()),
                             DepDirection::Same};
        }) |
        ranges::to<std::vector>()};

    PBCtx ctx;
    PBSet loop0Set, loop1Set;
    std::vector<IterAxis> outerAxes;

    auto extractLoopSet = [&](const AccessPoint &p) {
        auto iterList = AnalyzeDeps::makeIterList(p.iter_, p.iter_.size());
        GenPBExpr gen;
        GenPBExpr::VarMap externals;
        auto conds = *AnalyzeDeps::makeCond(gen, p.conds_, RelaxMode::Possible,
                                            externals);
        if (externals.size() > 0)
            ERROR("PlutoFuse: external variables currently "
                  "not supported.");

        PBSet loopSet(ctx, "{ " + iterList + ": " + conds + " }");

        // project out constant dims
        for (int64_t i = p.iter_.size() - 1; i >= 0; --i)
            if (p.iter_[i].realIter_->nodeType() != ASTNodeType::Var) {
                loopSet.projectOutDims(i, 1);
            }

        return loopSet;
    };

    auto findIter = [](const AccessPoint &ap, const std::string var) {
        std::vector<int> outerDims;
        outerDims.reserve(ap.iter_.size());
        for (auto &&[i, iterAxis] : views::enumerate(ap.iter_))
            if (iterAxis.realIter_->nodeType() == ASTNodeType::Var) {
                if (iterAxis.realIter_.as<VarNode>()->name_ == var)
                    return std::pair{i, std::move(outerDims)};
                else
                    outerDims.push_back(i);
            }
        ASSERT(false);
    };

    auto getDeps = [&](const For &l0, const For &l1,
                       bool handleFakeAccess = false) mutable {
        std::vector<PBSet> deps;
        FindDeps()
            .noProjectOutProvateAxis(true)
            .filterAccess([&](const AccessPoint &p) {
                if (p.def_->name_ == FAKE_ACCESS_VAR) {
                    if (handleFakeAccess) {
                        if (p.stmt_->ancestorById(loop0Id).isValid()) {
                            loop0Set = extractLoopSet(p);
                            outerAxes = p.iter_ |
                                        views::filter([](const IterAxis &axis) {
                                            return axis.realIter_->nodeType() ==
                                                   ASTNodeType::Var;
                                        }) |
                                        ranges::to<std::vector>();
                        } else {
                            ASSERT(p.stmt_->ancestorById(loop1Id).isValid());
                            loop1Set = extractLoopSet(p);
                        }
                    }
                    return false;
                }
                return true;
            })
            .filterEarlier([&](const AccessPoint &p) {
                return p.stmt_->ancestorById(l0->id()).isValid();
            })
            .filterLater([&](const AccessPoint &p) {
                return p.stmt_->ancestorById(l1->id()).isValid();
            })
            .direction(outersSame)(ast, [&](const Dependency &d) {
                // later to earlier map, but projects out unrelated dims
                auto hMap = d.later2EarlierIter_;

                if (hMap.nParamDims() > 0)
                    throw InvalidSchedule("PlutoFuse: load in loop ranges "
                                          "currently not supported.");

                // remove inner dims for outer
                auto [pos0, outerDims0] = findIter(d.earlier_, l0->iter_);
                pos0 += nestLevel;
                hMap.projectOutOutputDims(pos0, hMap.nOutDims() - pos0);
                pos0 -= nestLevel;
                for (int i = outerDims0.size() - 1; i >= 0;
                     pos0 = outerDims0[i--])
                    hMap.projectOutOutputDims(outerDims0[i] + 1,
                                              pos0 - outerDims0[i] - 1);
                hMap.projectOutOutputDims(0, pos0);

                // remove inner dims for later
                auto [pos1, outerDims1] = findIter(d.later_, l1->iter_);
                pos1 += nestLevel;
                hMap.projectOutInputDims(pos1, hMap.nInDims() - pos1);
                pos1 -= nestLevel;
                for (int i = outerDims1.size() - 1; i >= 0;
                     pos1 = outerDims1[i--])
                    hMap.projectOutInputDims(outerDims1[i] + 1,
                                             pos1 - outerDims1[i] - 1);
                hMap.projectOutInputDims(0, pos1);

                // flatten to set for later coefficients computation;
                // later dimensions first, so the first half would be target,
                // and second half being source
                auto hSet = flattenMapToSet(std::move(hMap));

                if (deps.size() > 0)
                    ASSERT(hSet.nDims() == deps[0].nDims());
                deps.emplace_back(ctx, toString(std::move(hSet)));
            });
        return deps;
    };

    int nParams = -1;
    auto printMapSource = [&,
                           mapSource = std::string()](int nParamsNew) mutable {
        // the Pluto "parameter"s are the outer dimensions for us.
        // their number should keep the same throughout this schedule.
        if (nParams != -1) {
            ASSERT(nParams == nParamsNew);
            return mapSource;
        }
        nParams = nParamsNew;

        std::ostringstream oss;
        oss << "{ [";
        // target params
        for (int i = 0; i < nParams; ++i) {
            if (i > 0)
                oss << ", ";
            oss << "tp" << i;
        }
        // target loops
        for (int i = 0; i < nestLevel; ++i) {
            if (i > 0 || nParams > 0)
                oss << ", ";
            oss << "ti" << i;
        }
        // source params
        for (int i = 0; i < nParams; ++i)
            oss << ", sp" << i;
        // source loops
        for (int i = 0; i < nestLevel; ++i)
            oss << ", si" << i;
        oss << "] -> [";
        return mapSource = oss.str();
    };

    // constraints for bounding and valid coefficients
    std::vector<PBSet> coeffSets0, coeffSets1, coeffSets1to0;
    // sets for coefficients strictly satisfying certain dependence
    std::vector<PBSet> satSets0, satSets1, satSets1to0;
    // whether a dependence have been satisfied already
    std::vector<bool> satisfied0, satisfied1, satisfied1to0;

    // Dependences inside loop 0
    auto dep0 = getDeps(loop0, loop0);
    if (dep0.size() > 0) {
        // ti (target iter) and si (source iter) both on loop 0.
        // legality problem:
        // c0 * ti - c0 * si >= 0
        // bounding problem:
        // c * parameters - (c0 * ti - c0 * si) >= 0
        // this part has no constraint on c1.
        PBMap legalityMap, boundingMap;
        const int nParams = dep0[0].nDims() / 2 - nestLevel;
        {
            std::ostringstream oss;

            // coefficients for bounds have no effect here
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            oss << ", 0";

            // loop 0 params coefficients, difference is always 0
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 0 constant coefficient, difference is always 0
            oss << ", 0";
            // loop 0 loops coefficients, difference
            for (int i = 0; i < nestLevel; ++i)
                oss << ", ti" << i << " - si" << i;

            // loop 1 params coefficients, no effect
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 1 constant coefficient, no effect
            oss << ", 0";
            // loop 1 loops coefficients, no effect
            for (int i = 0; i < nestLevel; ++i)
                oss << ", 0";

            oss << "] }";

            legalityMap =
                PBMap(ctx, printMapSource(nParams) + oss.str().substr(2));
        }
        {
            std::ostringstream oss;

            // repeat params first for bounding coefficients
            for (int i = 0; i < nParams; ++i)
                oss << ", sp" << i;
            // constant coefficient in bounding
            oss << ", 1";

            // loop 0 params coefficients, negated difference always 0
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 0 constant coefficient, negated difference always 0
            oss << ", 0";
            // loop 0 loops coefficients, negated difference
            for (int i = 0; i < nestLevel; ++i)
                oss << ", si" << i << " - ti" << i;

            // loop 1 params coefficients, no effect
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 1 constant coefficient, no effect
            oss << ", 0";
            // loop 1 loops coefficients, no effect
            for (int i = 0; i < nestLevel; ++i)
                oss << ", 0";

            oss << "] }";

            boundingMap =
                PBMap(ctx, printMapSource(nParams) + oss.str().substr(2));
        }
        // generate constraints
        coeffSets0 = dep0 | views::transform([&](const auto &d) {
                         return intersect(coefficients(apply(d, legalityMap)),
                                          coefficients(apply(d, boundingMap)));
                     }) |
                     ranges::to<std::vector>();
        satSets0 = dep0 | views::transform([&](const auto &d) {
                       return coefficients(apply(d, legalityMap), 1);
                   }) |
                   ranges::to<std::vector>();
        satisfied0.resize(dep0.size(), false);
    }

    // Dependences inside loop 1
    auto dep1 = getDeps(loop1, loop1);
    if (dep1.size() > 0) {
        // ti (target iter) and si (source iter) both on loop 1.
        // legality problem:
        // c1 * ti - c1 * si >= 0
        // bounding problem:
        // c * parameters - (c1 * ti - c1 * si) >= 0
        // this part has no constraint on c0.
        PBMap legalityMap, boundingMap;
        const int nParams = dep1[0].nDims() / 2 - nestLevel;
        {
            std::ostringstream oss;

            // coefficients for bounds have no effect here
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            oss << ", 0";

            // loop 0 params coefficients, no effect
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 0 constant coefficient, no effect
            oss << ", 0";
            // loop 0 loops coefficients, no effect
            for (int i = 0; i < nestLevel; ++i)
                oss << ", 0";

            // loop 1 params coefficients, difference is always 0
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 1 constant coefficient, difference is always 0
            oss << ", 0";
            // loop 1 loops coefficients, difference
            for (int i = 0; i < nestLevel; ++i)
                oss << ", ti" << i << " - si" << i;

            oss << "] }";

            legalityMap =
                PBMap(ctx, printMapSource(nParams) + oss.str().substr(2));
        }
        {
            std::ostringstream oss;

            // repeat params first for bounding coefficients
            for (int i = 0; i < nParams; ++i)
                oss << ", sp" << i;
            // constant coefficient in bounding
            oss << ", 1";

            // loop 0 params coefficients, no effect
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 0 constant coefficient, no effect
            oss << ", 0";
            // loop 0 loops coefficients, no effect
            for (int i = 0; i < nestLevel; ++i)
                oss << ", 0";

            // loop 1 params coefficients, negated difference always 0
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 1 constant coefficient, negated difference always 0
            oss << ", 0";
            // loop 1 loops coefficients, negated difference
            for (int i = 0; i < nestLevel; ++i)
                oss << ", si" << i << " - ti" << i;

            oss << "] }";

            boundingMap =
                PBMap(ctx, printMapSource(nParams) + oss.str().substr(2));
        }
        // generate constraints
        coeffSets1 = dep1 | views::transform([&](const auto &d) {
                         return intersect(coefficients(apply(d, legalityMap)),
                                          coefficients(apply(d, boundingMap)));
                     }) |
                     ranges::to<std::vector>();
        satSets1 = dep1 | views::transform([&](const auto &d) {
                       return coefficients(apply(d, legalityMap), 1);
                   }) |
                   ranges::to<std::vector>();
        satisfied1.resize(dep1.size(), false);
    }

    // Dependences between loop 0 and 1
    auto dep1to0 = getDeps(loop0, loop1, true);
    std::cout << "loop 0: " << loop0Set << std::endl;
    std::cout << "loop 1: " << loop1Set << std::endl;
    if (dep1to0.size() > 0) {
        // i0 = si (source iter)
        // i1 = ti (target iter)
        // legality problem:
        // c1 * i1 - c0 * i0 >= 0
        // bounding problem:
        // c * parameters - (c1 * i1 - c0 * i0) >= 0
        PBMap legalityMap, boundingMap;
        const int nParams = dep1to0[0].nDims() / 2 - nestLevel;
        {
            std::ostringstream oss;

            // coefficients for bounds have no effect here
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            oss << ", 0";

            // source params, negative
            for (int i = 0; i < nParams; ++i)
                oss << ", -sp" << i;
            // source constant, negative
            oss << ", -1";
            // source loops, negative
            for (int i = 0; i < nestLevel; ++i)
                oss << ", -si" << i;

            // target params, positive
            for (int i = 0; i < nParams; ++i)
                oss << ", tp" << i;
            // target constant, positive
            oss << ", 1";
            // target loops, positive
            for (int i = 0; i < nestLevel; ++i)
                oss << ", ti" << i;

            oss << "] }";

            legalityMap =
                PBMap(ctx, printMapSource(nParams) + oss.str().substr(2));
        }
        {
            std::ostringstream oss;

            // repeat params first for bounding coefficients
            for (int i = 0; i < nParams; ++i)
                oss << ", sp" << i;
            // constant coefficient in bounding
            oss << ", 1";

            // source params, positive
            for (int i = 0; i < nParams; ++i)
                oss << ", sp" << i;
            // target constant, positive
            oss << ", 1";
            // source loops, positive
            for (int i = 0; i < nestLevel; ++i)
                oss << ", si" << i;

            // target params, negative
            for (int i = 0; i < nParams; ++i)
                oss << ", -tp" << i;
            // target constant, negative
            oss << ", -1";
            // target loops, negative
            for (int i = 0; i < nestLevel; ++i)
                oss << ", -ti" << i;

            oss << "] }";

            boundingMap =
                PBMap(ctx, printMapSource(nParams) + oss.str().substr(2));
        }
        // generate constraints
        coeffSets1to0 =
            dep1to0 | views::transform([&](const auto &d) {
                return intersect(coefficients(apply(d, legalityMap)),
                                 coefficients(apply(d, boundingMap)));
            }) |
            ranges::to<std::vector>();
        satSets1to0 = dep1to0 | views::transform([&](const auto &d) {
                          return coefficients(apply(d, legalityMap), 1);
                      }) |
                      ranges::to<std::vector>();
        satisfied1to0.resize(dep1to0.size(), false);
    }

    //! FIXME: handle case of no dependence
    ASSERT(nParams != -1);

    // variable names
    auto cBound = [](int i) { return "cb" + toString(i); };

    const std::string c0Sum = "c0sum";
    const std::string delta0 = "d0";
    const std::string deltaL0 = "dl0";
    auto c0Param = [](int i) { return "c0p" + toString(i); };
    auto c0Iter = [](int i) { return "c0i" + toString(i); };
    auto c0 = [&](int i) {
        if (i < nParams + 1)
            return c0Param(i);
        else
            return c0Iter(i - nParams - 1);
    };

    const std::string c1Sum = "c1sum";
    const std::string delta1 = "d1";
    const std::string deltaL1 = "dl1";
    auto c1Param = [](int i) { return "c1p" + toString(i); };
    auto c1Iter = [](int i) { return "c1i" + toString(i); };
    auto c1 = [&](int i) {
        if (i < nParams + 1)
            return c1Param(i);
        else
            return c1Iter(i - nParams - 1);
    };

    // construct the map from coefficients to optimize targets
    PBMap optimizeMap;
    std::string optimizeTargets;
    {
        // the coefficients set includes following dimensions:
        // 1. (nParams + 1) bounding coefficients
        // 2. (nParams + 1 + nestLevel) loop 0 permuting coefficients
        // 3. (nParams + 1 + nestLevel) loop 1 permuting coefficients
        auto inputs = views::concat(
            views::ints(0, nParams + 1) | views::transform(cBound),
            views::ints(0, nParams + 1) | views::transform(c0Param),
            views::ints(0, nestLevel) | views::transform(c0Iter),
            views::ints(0, nParams + 1) | views::transform(c1Param),
            views::ints(0, nestLevel) | views::transform(c1Iter));

        // then the outputs which are optimize targets
        std::array extraOut0{c0Sum, delta0, deltaL0};
        std::array extraOut1{c1Sum, delta1, deltaL1};
        auto outputs = views::concat(
            // 1. the distance bounds go first, as main optimizing targets
            views::ints(0, nParams + 1) | views::transform(cBound),
            // 2.1. sum of coefficients absolute for loop 0
            //   and
            // 2.2. binary decision variable for avoiding zeros
            extraOut0,
            // 2.3. reversed coefficients of loop 0 iterations
            //      they are reversed because we want to select outer loops
            //      earlier, preserving the original loop order
            views::reverse(views::ints(0, nestLevel)) |
                views::transform(c0Iter),
            // 2.4. coefficients of loop 0 params and constant
            views::ints(0, nParams + 1) | views::transform(c0Param),
            // 3.1. sum of coefficients absolute for loop 1
            //   and
            // 3.2. binary decision variable for avoiding zeros
            extraOut1,
            // 3.3. reversed coefficients of loop 1 iterations
            views::reverse(views::ints(0, nestLevel)) |
                views::transform(c1Iter),
            // 3.4. coefficients of loop 1 params and constant
            views::ints(0, nParams + 1) | views::transform(c1Param));

        auto loopConstraints = [&](const auto &cSum, const auto &c,
                                   const auto &cIter, const auto &delta,
                                   const auto &deltaL) {
            auto absSum = absSumConstraints(nParams + 1 + nestLevel, cSum, c);
            auto nonZero = nonZeroConstraints(nestLevel, cIter, delta);
            std::array limitDeltaL{
                deltaL + " >= 0",
                deltaL + " <= 1",
            };
            return views::concat(absSum, nonZero, limitDeltaL) |
                   ranges::to<std::vector>();
        };

        // constraints excluding trivial results (all zero)
        auto constraints0 = loopConstraints(c0Sum, c0, c0Iter, delta0, deltaL0);
        auto constraints1 = loopConstraints(c1Sum, c1, c1Iter, delta1, deltaL1);
        auto constraints = views::concat(constraints0, constraints1);

        auto optimizeTargets = join(outputs, ", ");
        optimizeMap =
            PBMap(ctx, "{ [" + join(inputs, ", ") + "] -> [" + optimizeTargets +
                           "]: " + join(constraints, " and ") + "}");
    }
    auto orthoSetGen = [&](const auto &orthoConstraints) {
        return PBSet(ctx, "{ [" + optimizeTargets +
                              "]: " + join(orthoConstraints, " and ") + " }");
    };
    auto revOptimizeMap = reverse(optimizeMap);
    PBSet orthoSet = orthoSetGen(std::vector<std::string>{});

    std::vector<std::vector<int>> c0ParamFusedValue, c0IterFusedValue;
    std::vector<std::vector<int>> c1ParamFusedValue, c1IterFusedValue;
    // start computing permuted dimensions
    for (int i = 0; i < nestLevel; ++i) {
        //! FIXME: handle parameters from loads
        auto problem = universeSet(
            spaceSetAlloc(ctx, 0, (nParams + 1) * 3 + nestLevel * 2));
        // constructing the coefficients' space
        for (const auto &[satisfied, coeffSet] :
             views::zip(views::concat(satisfied0, satisfied1, satisfied1to0),
                        views::concat(coeffSets0, coeffSets1, coeffSets1to0)))
            if (!satisfied)
                problem = intersect(std::move(problem), coeffSet);
        // map the coefficients to optimize targets, and perform optimization
        auto solution =
            lexmin(intersect(std::move(orthoSet), apply(problem, optimizeMap)));
        if (solution.empty())
            break;
        auto solutionVals =
            sample(apply(std::move(solution), revOptimizeMap)).coordinates();
        auto optimized = solutionVals | views::transform([&](const PBVal &val) {
                             ASSERT(val.denSi() == 1);
                             return val.numSi();
                         }) |
                         ranges::to<std::vector>();
        std::cout << optimized << std::endl;

        // check and exclude fake fusion
        auto loopSetToRange = [&](int coeffBase) {
            return PBMap(
                ctx,
                "{ [" +
                    (views::concat(views::ints(0, nParams) |
                                       views::transform([](int i) {
                                           return "p" + toString(i);
                                       }),
                                   views::ints(0, nestLevel) |
                                       views::transform([](int i) {
                                           return "x" + toString(i);
                                       })) |
                     join(", ")) +
                    "] -> [" +
                    (views::concat(
                         views::ints(0, nParams + 1) |
                             views::transform([&](int i) {
                                 auto c = toString(optimized[coeffBase + i]);
                                 if (i == nParams)
                                     return c;
                                 else
                                     return c + "p" + toString(i);
                             }),
                         views::ints(0, nestLevel) |
                             views::transform([&](int i) {
                                 auto c = toString(
                                     optimized[coeffBase + nParams + 1 + i]);
                                 return c + "x" + toString(i);
                             })) |
                     join(" + ")) +
                    "] }");
        };
        PBSet loop0Range = apply(loop0Set, loopSetToRange(nParams + 1));
        PBSet loop1Range =
            apply(loop1Set, loopSetToRange((nParams + 1) * 2 + nestLevel));
        // if the two loops has no overlap on the result axis, they are not
        // actually fused so we bail out
        if (intersect(
                // range of values that less than the maximum of loop 0
                apply(lexmax(loop0Range), lexGE(spaceSetAlloc(ctx, 0, 1))),
                // ... and from loop 1
                loop1Range)
                // ... don't overlap
                .empty())
            break;

        // save coefficients' values
        c0ParamFusedValue.push_back({
            optimized.begin() + (nParams + 1),
            optimized.begin() + (nParams + 1) * 2,
        });
        c0IterFusedValue.push_back({
            optimized.begin() + (nParams + 1) * 2,
            optimized.begin() + (nParams + 1) * 2 + nestLevel,
        });
        c1ParamFusedValue.push_back({
            optimized.begin() + (nParams + 1) * 2 + nestLevel,
            optimized.begin() + (nParams + 1) * 3 + nestLevel,
        });
        c1IterFusedValue.push_back({
            optimized.begin() + (nParams + 1) * 3 + nestLevel,
            optimized.end(),
        });

        // prepare orthogonal constraints for next iteration
        auto orthoConstraints = [&](const auto &cIterFusedValue,
                                    const auto &cIter, const auto &deltaL) {
            auto ortho = orthogonalMatrix(cIterFusedValue);
            if (ortho.empty())
                return std::vector<std::string>{"true"};
            auto orthoDots = ortho | views::transform([&](const auto &o) {
                                 return "(" +
                                        (views::ints(0, nestLevel) |
                                         views::transform([&](int i) {
                                             return toString(o[i]) + cIter(i);
                                         }) |
                                         join(" + ")) +
                                        ")";
                             }) |
                             ranges::to<std::vector>();
            return nonZeroConstraints(
                ortho.size(), [&](int i) { return orthoDots[i]; }, deltaL);
        };
        auto ortho0 = orthoConstraints(c0IterFusedValue, c0Iter, deltaL0);
        auto ortho1 = orthoConstraints(c1IterFusedValue, c1Iter, deltaL1);
        orthoSet = orthoSetGen(views::concat(ortho0, ortho1));
    }

    return {ast, {}};
}

} // namespace freetensor
