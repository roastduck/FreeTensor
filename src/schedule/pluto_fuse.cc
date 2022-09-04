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
                deps.emplace_back(ctx, toString(flattenMapToSet(hMap)));
            });
        return deps;
    };

    auto printMapSource = [&, nParamsOld = -1](std::ostringstream &oss,
                                               int nParams) mutable {
        // the Pluto "parameter"s are the outer dimensions for us.
        // their number should keep the same throughout this schedule.
        if (nParamsOld != -1)
            ASSERT(nParamsOld == nParams);
        else
            nParamsOld = nParams;

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
    };

    std::vector<PBSet> coeffSets0, coeffSets1, coeffSets1to0;

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
            printMapSource(oss, nParams);

            // coefficients for bounds have no effect here
            for (int i = 0; i < nParams; ++i) {
                if (i > 0)
                    oss << ", ";
                oss << "0";
            }
            // loop 0 params coefficients, difference is always 0
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 0 loops coefficients, difference
            for (int i = 0; i < nestLevel; ++i) {
                if (i > 0 || nParams > 0)
                    oss << ", ";
                oss << "ti" << i << " - si" << i;
            }
            // loop 1 params coefficients, no effect
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 1 loops coefficients, no effect
            for (int i = 0; i < nestLevel; ++i)
                oss << ", 0";
            oss << "] }";
            legalityMap = PBMap(ctx, oss.str());
        }
        {
            std::ostringstream oss;
            printMapSource(oss, nParams);

            // repeat params first for bounding coefficients
            for (int i = 0; i < nParams; ++i) {
                if (i > 0)
                    oss << ", ";
                oss << "sp" << i;
            }
            // loop 0 params coefficients, negated difference always 0
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 0 loops coefficients, negated difference
            for (int i = 0; i < nestLevel; ++i) {
                if (i > 0 || nParams > 0)
                    oss << ", ";
                oss << "si" << i << " - ti" << i;
            }
            // loop 1 params coefficients, no effect
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 1 loops coefficients, no effect
            for (int i = 0; i < nestLevel; ++i)
                oss << ", 0";
            oss << "] }";
            boundingMap = PBMap(ctx, oss.str());
        }
        // generate constraints
        coeffSets0 = dep0 | iter::imap([&](const auto &d) {
                         return intersect(coefficients(apply(d, legalityMap)),
                                          coefficients(apply(d, boundingMap)));
                     }) |
                     collect;
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
            printMapSource(oss, nParams);

            // coefficients for bounds have no effect here
            for (int i = 0; i < nParams; ++i) {
                if (i > 0)
                    oss << ", ";
                oss << "0";
            }
            // loop 0 params coefficients, no effect
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 0 loops coefficients, no effect
            for (int i = 0; i < nestLevel; ++i) {
                if (i > 0 || nParams > 0)
                    oss << ", ";
                oss << "0";
            }
            // loop 1 params coefficients, difference is always 0
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 1 loops coefficients, difference
            for (int i = 0; i < nestLevel; ++i)
                oss << ", ti" << i << " - si" << i;
            oss << "] }";
            legalityMap = PBMap(ctx, oss.str());
        }
        {
            std::ostringstream oss;
            printMapSource(oss, nParams);

            // repeat params first for bounding coefficients
            for (int i = 0; i < nParams; ++i) {
                if (i > 0)
                    oss << ", ";
                oss << "sp" << i;
            }
            // loop 0 params coefficients, no effect
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 0 loops coefficients, no effect
            for (int i = 0; i < nestLevel; ++i) {
                if (i > 0 || nParams > 0)
                    oss << ", ";
                oss << "0";
            }
            // loop 1 params coefficients, negated difference always 0
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 1 loops coefficients, negated difference
            for (int i = 0; i < nestLevel; ++i)
                oss << ", si" << i << " - ti" << i;
            oss << "] }";
            boundingMap = PBMap(ctx, oss.str());
        }
        // generate constraints
        coeffSets1 = dep1 | iter::imap([&](const auto &d) {
                         return intersect(coefficients(apply(d, legalityMap)),
                                          coefficients(apply(d, boundingMap)));
                     }) |
                     collect;
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
            printMapSource(oss, nParams);

            // coefficients for bounds have no effect here
            for (int i = 0; i < nParams; ++i) {
                if (i > 0)
                    oss << ", ";
                oss << "0";
            }
            // source params, negative
            for (int i = 0; i < nParams; ++i)
                oss << ", -sp" << i;
            // source loops, negative
            for (int i = 0; i < nestLevel; ++i) {
                if (i > 0 || nParams > 0)
                    oss << ", ";
                oss << "-si" << i;
            }
            // target params, negative
            for (int i = 0; i < nParams; ++i)
                oss << ", tp" << i;
            // target loops, negative
            for (int i = 0; i < nestLevel; ++i)
                oss << ", ti" << i;
            oss << "] }";
            legalityMap = PBMap(ctx, oss.str());
        }
        {
            std::ostringstream oss;
            printMapSource(oss, nParams);

            // repeat params first for bounding coefficients
            for (int i = 0; i < nParams; ++i) {
                if (i > 0)
                    oss << ", ";
                oss << "sp" << i;
            }
            // source params, positive
            for (int i = 0; i < nParams; ++i)
                oss << ", sp" << i;
            // source loops, positive
            for (int i = 0; i < nestLevel; ++i) {
                if (i > 0 || nParams > 0)
                    oss << ", ";
                oss << "si" << i;
            }
            // target params, negative
            for (int i = 0; i < nParams; ++i)
                oss << ", -tp" << i;
            // target loops, negative
            for (int i = 0; i < nestLevel; ++i)
                oss << ", -ti" << i;
            oss << "] }";
            boundingMap = PBMap(ctx, oss.str());
        }
        // generate constraints
        coeffSets1to0 =
            dep1to0 | iter::imap([&](const auto &d) {
                return intersect(coefficients(apply(d, legalityMap)),
                                 coefficients(apply(d, boundingMap)));
            }) |
            collect;
    }

    std::cout << coeffSets0 << std::endl;
    std::cout << coeffSets1 << std::endl;
    std::cout << coeffSets1to0 << std::endl;

    return {ast, {}};
}

} // namespace freetensor
