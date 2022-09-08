#include <cmath>
#include <numeric>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <pass/flatten_stmt_seq.h>
#include <schedule/fuse.h>
#include <schedule/pluto_fuse.h>

namespace freetensor {

std::pair<Stmt, ID> plutoFuse(const Stmt &_ast, const ID &loop0,
                              const ID &loop1, int nestLevel) {
    // flatten first so we get perfectly nested loops as much as possible
    auto ast = flattenStmtSeq(_ast);

    // check accessed vardefs: those vardefs accessed by loop1 should not have
    // their shapes modified in loop0
    CheckFuseAccessible check(loop0, loop1);
    check.check(ast);

    // count maximum count of perfectly nested loops at loop0 and loop1
    auto countPerfectNest = [](const For &loop) {
        int n = 0;
        for (Stmt inner = loop; inner->nodeType() == ASTNodeType::For;
             inner = inner.as<ForNode>()->body_)
            n++;
        return n;
    };
    int nestLevel0 = countPerfectNest(check.loop0().loop_);
    int nestLevel1 = countPerfectNest(check.loop1().loop_);

    // Process nestLevel nested loops each side; default to
    if (nestLevel == -1)
        nestLevel = std::min(nestLevel0, nestLevel1);
    else if (nestLevel0 < nestLevel)
        throw InvalidSchedule(
            "PlutoFuse: loop 0 `#" + toString(loop0) +
            "` has less than required nesting levels: " + toString(nestLevel0) +
            " existed, but " + toString(nestLevel) + " required");
    else if (nestLevel1 < nestLevel)
        throw InvalidSchedule(
            "PlutoFuse: loop 1 `#" + toString(loop1) +
            "` has less than required nesting levels: " + toString(nestLevel1) +
            " existed, but " + toString(nestLevel) + " required");

    ASSERT(nestLevel > 0);

    std::cout << "nesting levels = " << nestLevel << std::endl;

    // List common outer loops
    std::deque<For> outers;
    for (Stmt outer = check.loop0().loop_->parentStmt(); outer.isValid();
         outer = outer->parentStmt())
        if (outer->nodeType() == ASTNodeType::For)
            outers.push_front(outer.as<ForNode>());

    // Sanity check: the two loops should have the same outer loops given the
    // CheckFuseAccessible passed; just check if the innermost one aligns
    auto loop1InnermostOuter = check.loop1().loop_->parentStmtByFilter(
        [](const Stmt &s) { return s->nodeType() == ASTNodeType::For; });
    // outers are empty, loop1 should also have no outer
    if (outers.empty())
        ASSERT(!loop1InnermostOuter.isValid());
    // outers exist, the innermost outer of loop1 should be the same as loop0
    else
        ASSERT(loop1InnermostOuter == outers.back());

    std::vector<FindDepsDir> outersSame{outers | iter::imap([&](const For &f) {
                                            return std::pair{
                                                NodeIDOrParallelScope(f->id()),
                                                DepDirection::Same};
                                        }) |
                                        collect};

    PBCtx ctx;

    auto findIter = [](const AccessPoint &ap, const std::string var) {
        std::vector<int> outerDims;
        outerDims.reserve(ap.iter_.size());
        for (auto &&[i, iterAxis] : iter::enumerate(ap.iter_))
            if (iterAxis.realIter_->nodeType() == ASTNodeType::Var) {
                if (iterAxis.realIter_.as<VarNode>()->name_ == var)
                    return std::pair{i, std::move(outerDims)};
                else
                    outerDims.push_back(i);
            }
        ASSERT(false);
    };

    auto getDeps = [&](const For &l0, const For &l1) mutable {
        std::vector<PBSet> deps;
        FindDeps()
            .noProjectOutProvateAxis(true)
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

                std::cout << l0->metadata() << " -> " << l1->metadata() << ": "
                          << hMap << std::endl;

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
    auto dep0 = getDeps(check.loop0().loop_, check.loop0().loop_);
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
        coeffSets0 = dep0 | iter::imap([&](const auto &d) {
                         return intersect(coefficients(apply(d, legalityMap)),
                                          coefficients(apply(d, boundingMap)));
                     }) |
                     collect;
        satSets0 = dep0 | iter::imap([&](const auto &d) {
                       return coefficients(apply(d, legalityMap), 1);
                   }) |
                   collect;
        satisfied0.resize(dep0.size(), false);
    }

    // Dependences inside loop 1
    auto dep1 = getDeps(check.loop1().loop_, check.loop1().loop_);
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
        coeffSets1 = dep1 | iter::imap([&](const auto &d) {
                         return intersect(coefficients(apply(d, legalityMap)),
                                          coefficients(apply(d, boundingMap)));
                     }) |
                     collect;
        satSets1 = dep1 | iter::imap([&](const auto &d) {
                       return coefficients(apply(d, legalityMap), 1);
                   }) |
                   collect;
        satisfied1.resize(dep1.size(), false);
    }

    // Dependences between loop 0 and 1
    auto dep1to0 = getDeps(check.loop0().loop_, check.loop1().loop_);
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
            dep1to0 | iter::imap([&](const auto &d) {
                return intersect(coefficients(apply(d, legalityMap)),
                                 coefficients(apply(d, boundingMap)));
            }) |
            collect;
        satSets1to0 = dep1to0 | iter::imap([&](const auto &d) {
                          return coefficients(apply(d, legalityMap), 1);
                      }) |
                      collect;
        satisfied1to0.resize(dep1to0.size(), false);
    }

    //! FIXME: handle case of no dependence
    ASSERT(nParams != -1);

    // construct the map from coefficients to optimize targets
    PBMap optimizeMap;
    {
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

        // the coefficients set includes following dimensions:
        // 1. (nParams + 1) bounding coefficients
        // 2. (nParams + 1 + nestLevel) loop 0 permuting coefficients
        // 3. (nParams + 1 + nestLevel) loop 1 permuting coefficients
        auto inputs =
            iter::chain(iter::range(nParams + 1) | iter::imap(cBound),
                        iter::range(nParams + 1) | iter::imap(c0Param),
                        iter::range(nestLevel) | iter::imap(c0Iter),
                        iter::range(nParams + 1) | iter::imap(c1Param),
                        iter::range(nestLevel) | iter::imap(c1Iter)) |
            collect;

        // then the outputs which are optimize targets
        auto outputs =
            iter::chain(
                // 1. the distance bounds go first, as main optimizing targets
                iter::range(nParams + 1) | iter::imap(cBound) | collect,
                // 2.1. sum of coefficients absolute for loop 0
                std::array{c0Sum},
                // 2.2. binary decision variable for avoiding zeros
                std::array{delta0, deltaL0},
                // 2.3. reversed coefficients of loop 0 iterations
                //      they are reversed because we want to select outer loops
                //      earlier, preserving the original loop order
                iter::range(nestLevel - 1, -1, -1) | iter::imap(c0Iter) |
                    collect,
                // 2.4. coefficients of loop 0 params and constant
                iter::range(nParams + 1) | iter::imap(c0Param) | collect,
                // 3.1. sum of coefficients absolute for loop 1
                std::array{c1Sum},
                // 3.2. binary decision variable for avoiding zeros
                std::array{delta1, deltaL1},
                // 3.3. reversed coefficients of loop 1 iterations
                iter::range(nestLevel - 1, -1, -1) | iter::imap(c1Iter) |
                    collect,
                // 3.4. coefficients of loop 1 params and constant
                iter::range(nParams + 1) | iter::imap(c1Param) | collect) |
            collect;

        auto sumConstraints = [&](const auto &cSum, const auto &c) {
            return iter::range(nParams + 1 + nestLevel) | iter::powerset |
                   iter::imap([&](const auto &v) {
                       std::ostringstream oss;
                       oss << cSum << " >= 0";
                       size_t iv = 0;
                       for (int i = 0; i < nParams + 1 + nestLevel; ++i) {
                           if (iv < v.size() && i == v[iv]) {
                               iv++;
                               oss << " + ";
                           } else
                               oss << " - ";
                           oss << c(i);
                       }
                       return oss.str();
                   }) |
                   collect;
        };

        auto iterNonZeroConstraints = [&](const auto &cIter, const auto &delta,
                                          const auto &deltaL) {
            auto rangeConstraints =
                iter::range(nestLevel) | iter::imap([&](int i) {
                    return std::array{cIter(i) + " >= -2", cIter(i) + " <= 2"};
                }) |
                iter::chain.from_iterable | collect;
            auto skewedExpr =
                iter::range(nestLevel) | iter::imap([&](int i) {
                    return toString((int)std::pow(5, i)) + cIter(i);
                }) |
                join(" + ");
            auto bound = (int)std::pow(5, nestLevel);
            return iter::chain(
                       rangeConstraints,
                       std::array{
                           skewedExpr + " >= 1 - " + toString(bound) + delta,
                           skewedExpr + " <= " + toString(bound - 1) + " - " +
                               toString(bound) + delta,
                           delta + " >= 0",
                           delta + " <= 1",
                           deltaL + " >= 0",
                           deltaL + " <= 1",
                       }) |
                   collect;
        };

        // constraints excluding trivial results (all zero)
        auto constraints =
            iter::chain(sumConstraints(c0Sum, c0), sumConstraints(c1Sum, c1),
                        iterNonZeroConstraints(c0Iter, delta0, deltaL0),
                        iterNonZeroConstraints(c1Iter, delta1, deltaL1)) |
            collect;

        optimizeMap = PBMap(ctx, "{ [" + join(inputs, ", ") + "] -> [" +
                                     join(outputs, ", ") +
                                     "]: " + join(constraints, " and ") + "}");
        std::cout << optimizeMap << std::endl;
    }

    // start computing permuted dimensions
    for (int i = 0; i < nestLevel; ++i) {
        //! FIXME: handle parameters from loads
        auto problem = universeSet(
            spaceSetAlloc(ctx, 0, (nParams + 1) * 3 + nestLevel * 2));
        // constructing the coefficients' space
        for (const auto &c : iter::chain(coeffSets0, coeffSets1, coeffSets1to0))
            problem = intersect(std::move(problem), c);
        std::cout << problem << std::endl;
        // map the coefficients to optimize targets
        std::cout << lexmin(apply(problem, optimizeMap)) << std::endl;
        //! FIXME: compute linear independence constraints
        break;
    }

    return {ast, {}};
}

} // namespace freetensor
