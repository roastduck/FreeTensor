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
        for (auto inner = loop; inner->body_->nodeType() == ASTNodeType::For;
             inner = inner->body_.as<ForNode>())
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

    // List common outer loops
    int nOuter = 0;
    for (Stmt outer = check.loop0().loop_; outer->parentStmt().isValid();
         outer = outer->parentStmt())
        if (outer->nodeType() == ASTNodeType::For)
            nOuter++;
    std::vector<For> outers(nOuter);
    for (Stmt outer = check.loop0().loop_; outer->parentStmt().isValid();
         outer = outer->parentStmt())
        if (outer->nodeType() == ASTNodeType::For)
            outers[nOuter--] = outer.as<ForNode>();
    // Sanity check: the two loops should have the same outer loops given the
    // CheckFuseAccessible passed; just check if the innermost one aligns
    ASSERT(check.loop1().loop_->parentStmtByFilter([](const Stmt &s) {
        return s->nodeType() == ASTNodeType::For;
    }) == outers.back());
    std::vector<FindDepsDir> outersSame{outers | iter::imap([&](const For &f) {
                                            return std::pair{
                                                NodeIDOrParallelScope(f->id()),
                                                DepDirection::Same};
                                        }) |
                                        collect};

    PBCtx ctx;

    auto findIter = [](const AccessPoint &ap, const std::string var) {
        for (auto &&[i, iterAxis] : iter::enumerate(ap.iter_))
            if (iterAxis.realIter_->nodeType() == ASTNodeType::Var &&
                iterAxis.realIter_.as<VarNode>()->name_ == var)
                return i;
        ASSERT(false);
    };

    auto getDeps = [&, iterPos = -1](const For &l0, const For &l1) mutable {
        std::vector<PBSet> deps;
        FindDeps()
            .filterEarlier([&](const AccessPoint &p) {
                return p.stmt_->ancestorById(l0->id()).isValid();
            })
            .filterLater([&](const AccessPoint &p) {
                return p.stmt_->ancestorById(l1->id()).isValid();
            })
            .direction(outersSame)(ast, [&](const Dependency &d) {
                // later to earlier map, but projects out unrelated dims
                auto hMap = d.later2EarlierIter_;
                // remove inner input dims
                auto iter0Pos = findIter(d.later_, l0->iter_) + nestLevel;
                hMap.projectOutInputDims(iter0Pos, hMap.nInDims() - iter0Pos);
                auto iter1Pos = findIter(d.earlier_, l1->iter_) + nestLevel;
                hMap.projectOutOutputDims(iter1Pos, hMap.nOutDims() - iter1Pos);

                // sanity check: outer iterators should always be the same
                ASSERT(iter0Pos == iter1Pos);
                if (iterPos != -1)
                    ASSERT(iterPos == (int)iter0Pos);
                else
                    iterPos = iter0Pos;

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
        // source params
        for (int i = 0; i < nParams; ++i) {
            if (i > 0)
                oss << ", ";
            oss << "sp" << i;
        }
        // source loops
        for (int i = 0; i < nestLevel; ++i) {
            if (i > 0 || nParams > 0)
                oss << ", ";
            oss << "si" << i;
        }
        // target params
        for (int i = 0; i < nParams; ++i)
            oss << ", tp" << i;
        // target loops
        for (int i = 0; i < nestLevel; ++i)
            oss << ", ti" << i;
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
                oss << "ti - si" << i;
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
                oss << "si - ti" << i;
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
                oss << ", 0";
            }
            // loop 1 params coefficients, difference is always 0
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 1 loops coefficients, difference
            for (int i = 0; i < nestLevel; ++i)
                oss << "ti - si" << i;
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
                oss << ", 0";
            }
            // loop 1 params coefficients, negated difference always 0
            for (int i = 0; i < nParams; ++i)
                oss << ", 0";
            // loop 1 loops coefficients, negated difference
            for (int i = 0; i < nestLevel; ++i)
                oss << "si - ti" << i;
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
