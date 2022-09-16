#include <cmath>
#include <numeric>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <schedule/fuse.h>
#include <schedule/pluto_fuse.h>

namespace freetensor {

namespace {

PBBuildExpr absSumConstraint(const PBBuildExpr &sumVar,
                             const std::vector<PBBuildExpr> &vars) {
    auto n = vars.size();
    PBBuildExpr ret = true;
    ASSERT(n < 31);
    for (int i = 0; i < (1 << n); ++i) {
        PBBuildExpr absSum = 0;
        for (unsigned bit = 0; bit < n; ++bit) {
            if ((i >> bit) & 1)
                absSum += vars[bit];
            else
                absSum -= vars[bit];
        }
        ret = ret && sumVar >= absSum;
    }
    return ret;
}

PBBuildExpr nonZeroConstraint(const std::vector<PBBuildExpr> &vars,
                              const PBBuildExpr &delta, int varBound = 2) {
    auto n = vars.size();
    PBBuildExpr ret = true;
    if (n > 0u) {
        for (const auto &var : vars)
            ret = ret && var >= -varBound && var <= varBound;

        PBBuildExpr skewedExpr = 0;
        for (unsigned i = 0; i < n; ++i) {
            skewedExpr += int(std::pow(varBound * 2 + 1, i)) * vars[i];
        }

        auto bound = (int)std::pow(varBound * 2 + 1, n);
        ret = ret && skewedExpr >= 1 - bound * delta;
        ret = ret && skewedExpr <= bound - 1 - bound * delta;
    }
    return ret && delta >= 0 && delta <= 1;
}

std::vector<std::vector<int>>
orthogonalMatrix(const std::vector<std::vector<int>> &vectors) {
    // sanity check
    ASSERT(vectors.size() > 0);
    for (size_t i = 1; i < vectors.size(); ++i)
        ASSERT(vectors[i].size() == vectors[0].size());
    int nDims = vectors[0].size();

    PBSetBuilder builder;
    auto sum = builder.newVar("sum");
    auto vars = builder.newVars(nDims, "x");
    auto delta = builder.newVar("delta");
    auto fOrtho = [&](const std::vector<int> &coeffs) {
        PBBuildExpr ortho = 0;
        for (auto &&[c, x] : views::zip(coeffs, vars))
            ortho += c * x;
        return ortho == 0;
    };

    PBCtx ctx;
    builder.addConstraint(nonZeroConstraint(vars, delta));
    builder.addConstraint(absSumConstraint(sum, vars));
    builder.addConstraints(vectors | views::transform(fOrtho));
    PBSet orthogonalSet = builder.build(ctx);
    builder.clearConstraints();

    std::vector<std::vector<int>> result;
    while (!orthogonalSet.empty()) {
        // lexmin and sample to solve
        auto solution = sample(lexmin(orthogonalSet)).coordinates();
        // extract the vector into result
        std::vector<int> v;
        v.reserve(solution.size() - 2);
        for (size_t i = 1; i < solution.size() - 1; ++i) {
            ASSERT(solution[i].denSi() == 1);
            // put negated value into v since the result is minimized but we
            // want positive results
            v.emplace_back(-solution[i].numSi());
        }
        // inject new constraint to find next orthogonal vector
        builder.addConstraint(fOrtho(v));
        result.emplace_back(std::move(v));
        orthogonalSet = intersect(std::move(orthogonalSet), builder.build(ctx));
        builder.clearConstraints();
    }

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

PBSet extractLoopSet(const PBCtx &ctx, const AccessPoint &p) {
    auto iterList = AnalyzeDeps::makeIterList(p.iter_, p.iter_.size());
    GenPBExpr gen;
    GenPBExpr::VarMap externals;
    auto conds =
        *AnalyzeDeps::makeCond(gen, p.conds_, RelaxMode::Possible, externals);
    if (externals.size() > 0)
        ERROR("PlutoFuse: external variables currently "
              "not supported.");

    PBSet loopSet(ctx, "{ " + iterList + ": " + conds + " }");

    // project out constant dims
    for (int64_t i = p.iter_.size() - 1; i >= 0; --i)
        if (p.iter_[i].realIter_->nodeType() != ASTNodeType::Var)
            loopSet.projectOutDims(i, 1);

    return loopSet;
}

std::vector<IterAxis> extractVarAxes(const std::vector<IterAxis> &axes,
                                     const For &targetLoop) {
    std::vector<IterAxis> ret;
    for (auto &&axis : axes)
        if (axis.realIter_->nodeType() == ASTNodeType::Var) {
            if (axis.realIter_.as<VarNode>()->name_ == targetLoop->iter_)
                return ret;
            ret.emplace_back(axis);
        }
    ASSERT(false && "PlutoFuse: target loop is not found as expected");
}

std::pair<int, std::vector<int>> findIterFromAP(const AccessPoint &ap,
                                                const std::string var) {
    std::vector<int> outerDims;
    int n = ap.iter_.size();
    outerDims.reserve(n);
    for (int i = 0; i < n; ++i)
        if (ap.iter_[i].realIter_->nodeType() == ASTNodeType::Var) {
            if (ap.iter_[i].realIter_.as<VarNode>()->name_ == var)
                return std::pair{i, std::move(outerDims)};
            else
                outerDims.push_back(i);
        }
    ASSERT(false);
}

} // namespace

std::pair<Stmt, ID> plutoFuse(const Stmt &_ast, const ID &loop0Id,
                              const ID &loop1Id) {
    // flatten first so we get perfectly nested loops as much as possible
    auto original_ast = flattenStmtSeq(hoistVarOverStmtSeq(_ast));

    // check accessed vardefs: those vardefs accessed by loop1 should not have
    // their shapes modified in loop0
    CheckFuseAccessible check(loop0Id, loop1Id);
    check.check(original_ast);

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
    auto ast = original_ast;
    ast = InjectFakeAccess(inner0)(ast);
    ast = InjectFakeAccess(inner1)(ast);

    auto loop0 = findStmt(ast, loop0Id).as<ForNode>();
    auto loop1 = findStmt(ast, loop1Id).as<ForNode>();

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

    std::vector<FindDepsDir> outersSame{{}};
    for (auto &&f : outers)
        outersSame[0].emplace_back(f->id(), DepDirection::Same);

    PBCtx ctx;
    PBSet loop0Set, loop1Set;
    std::vector<IterAxis> outerAxes;

    auto getDeps = [&](const For &l0, int n0, const For &l1, int n1,
                       bool handleFakeAccess = false) mutable {
        std::vector<PBSet> deps;
        FindDeps()
            .noProjectOutProvateAxis(true)
            .filterAccess([&](const AccessPoint &p) {
                if (p.def_->name_ == FAKE_ACCESS_VAR) {
                    if (handleFakeAccess) {
                        if (p.stmt_->ancestorById(loop0Id).isValid()) {
                            loop0Set = extractLoopSet(ctx, p);
                            outerAxes = extractVarAxes(p.iter_, loop0);
                        } else {
                            ASSERT(p.stmt_->ancestorById(loop1Id).isValid());
                            loop1Set = extractLoopSet(ctx, p);
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
                auto [pos0, outerDims0] = findIterFromAP(d.earlier_, l0->iter_);
                pos0 += n0;
                hMap.projectOutOutputDims(pos0, hMap.nOutDims() - pos0);
                pos0 -= n0;
                for (int i = outerDims0.size() - 1; i >= 0;
                     pos0 = outerDims0[i--])
                    hMap.projectOutOutputDims(outerDims0[i] + 1,
                                              pos0 - outerDims0[i] - 1);
                hMap.projectOutOutputDims(0, pos0);

                // remove inner dims for later
                auto [pos1, outerDims1] = findIterFromAP(d.later_, l1->iter_);
                pos1 += n1;
                hMap.projectOutInputDims(pos1, hMap.nInDims() - pos1);
                pos1 -= n1;
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

    // find dependences
    auto dep0 = getDeps(loop0, nestLevel0, loop0, nestLevel0);
    auto dep1 = getDeps(loop1, nestLevel1, loop1, nestLevel1);
    auto dep1to0 = getDeps(loop0, nestLevel0, loop1, nestLevel1, true);
    int nParams = outerAxes.size();

    // constraints for bounding and valid coefficients
    std::vector<PBSet> coeffSets0, coeffSets1, coeffSets1to0;
    // sets for coefficients strictly satisfying certain dependence
    std::vector<PBSet> satSets0, satSets1, satSets1to0;
    // whether a dependence have been satisfied already
    std::vector<bool> satisfied0, satisfied1, satisfied1to0;

    // process dependences inside loop 0
    if (!dep0.empty()) {
        // ti (target iter) and si (source iter) both on loop 0.
        // this part has no constraint on c1.
        PBMapBuilder builder;
        auto tp = builder.newInputs(nParams, "tp");
        auto ti = builder.newInputs(nestLevel0, "ti");
        auto sp = builder.newInputs(nParams, "sp");
        auto si = builder.newInputs(nestLevel0, "si");

        // legality problem:
        // c0 * ti - c0 * si >= 0
        // bounds and loop 0 param coefficients, difference is always 0
        builder.addOutputs(views::repeat_n(0, (nParams + 1) * 2));
        // loop 0 loops coefficients, difference of iters
        builder.addOutputs(views::zip_with(
            [](auto &&ti, auto &&si) { return ti - si; }, ti, si));
        // loop 1 no effect
        builder.addOutputs(views::repeat_n(0, nParams + 1 + nestLevel1));
        auto legalityMap = builder.build(ctx);
        builder.clearOutputs();

        // bounding problem:
        // c * parameters - (c0 * ti - c0 * si) >= 0
        // bounds coefficients applied to params, use sp (should have sp===tp)
        builder.addOutputs(sp);
        builder.addOutput(1);
        // loop 0 param coefficients, difference is always 0
        builder.addOutputs(views::repeat_n(0, nParams + 1));
        // loop 0 loops coefficients, negated difference of iters
        builder.addOutputs(views::zip_with(
            [](auto &&ti, auto &&si) { return -(ti - si); }, ti, si));
        // loop 1 no effect
        builder.addOutputs(views::repeat_n(0, nParams + 1 + nestLevel1));
        auto boundingMap = builder.build(ctx);

        coeffSets0.reserve(dep0.size());
        satSets0.reserve(dep0.size());
        satisfied0.resize(dep0.size(), false);
        for (const auto &d : dep0) {
            coeffSets0.push_back(
                intersect(coefficients(apply(d, legalityMap)),
                          coefficients(apply(d, boundingMap))));
            satSets0.push_back(coefficients(apply(d, legalityMap), 1));
        }
    }

    // process dependences inside loop 1
    if (!dep1.empty()) {
        // ti (target iter) and si (source iter) both on loop 1.
        // this part has no constraint on c0.
        PBMapBuilder builder;
        auto tp = builder.newInputs(nParams, "tp");
        auto ti = builder.newInputs(nestLevel1, "ti");
        auto sp = builder.newInputs(nParams, "sp");
        auto si = builder.newInputs(nestLevel1, "si");

        // legality problem:
        // c1 * ti - c1 * si >= 0
        // bounds coefficients, no effect
        builder.addOutputs(views::repeat_n(0, nParams + 1));
        // loop 0 no effect
        builder.addOutputs(views::repeat_n(0, nParams + 1 + nestLevel0));
        // loop 1 param coefficients, difference is always 0
        builder.addOutputs(views::repeat_n(0, nParams + 1));
        // loop 1 loops coefficients, difference of iters
        builder.addOutputs(views::zip_with(
            [](auto &&ti, auto &&si) { return ti - si; }, ti, si));
        auto legalityMap = builder.build(ctx);
        builder.clearOutputs();

        // bounding problem:
        // c * parameters - (c1 * ti - c1 * si) >= 0
        // bounds coefficients applied to params, use sp (should have sp===tp)
        builder.addOutputs(sp);
        builder.addOutput(1);
        // loop 0 no effect
        builder.addOutputs(views::repeat_n(0, nParams + 1 + nestLevel0));
        // loop 1 param coefficients, difference is always 0
        builder.addOutputs(views::repeat_n(0, nParams + 1));
        // loop 1 loops coefficients, negated difference of iters
        builder.addOutputs(views::zip_with(
            [](auto &&ti, auto &&si) { return -(ti - si); }, ti, si));
        auto boundingMap = builder.build(ctx);

        coeffSets1.reserve(dep1.size());
        satSets1.reserve(dep1.size());
        satisfied1.resize(dep1.size(), false);
        for (const auto &d : dep1) {
            coeffSets1.push_back(
                intersect(coefficients(apply(d, legalityMap)),
                          coefficients(apply(d, boundingMap))));
            satSets1.push_back(coefficients(apply(d, legalityMap), 1));
        }
    }

    // Dependences between loop 0 and 1
    if (!dep1to0.empty()) {
        // i0 = si (source iter)
        // i1 = ti (target iter)
        PBMapBuilder builder;
        auto p0 = builder.newInputs(nParams, "p0");
        auto i0 = builder.newInputs(nestLevel0, "i0");
        auto p1 = builder.newInputs(nParams, "p1");
        auto i1 = builder.newInputs(nestLevel1, "i1");

        auto negate = views::transform([](auto &&x) { return -x; });

        // legality problem:
        // c1 * i1 - c0 * i0 >= 0
        // bounds coefficients, no effect
        builder.addOutputs(views::repeat_n(0, nParams + 1));
        // c0
        builder.addOutputs(p0 | negate);
        builder.addOutput(-1);
        builder.addOutputs(i0 | negate);
        // c1
        builder.addOutputs(p1);
        builder.addOutput(1);
        builder.addOutputs(i1);
        auto legalityMap = builder.build(ctx);
        builder.clearOutputs();

        // bounding problem:
        // c * parameters - (c1 * i1 - c0 * i0) >= 0
        // bounds coefficients
        builder.addOutputs(p0);
        builder.addOutput(1);
        // c0
        builder.addOutputs(p0);
        builder.addOutput(1);
        builder.addOutputs(i0);
        // c1
        builder.addOutputs(p1 | negate);
        builder.addOutput(-1);
        builder.addOutputs(i1 | negate);
        auto boundingMap = builder.build(ctx);

        coeffSets1to0.reserve(dep1to0.size());
        satSets1to0.reserve(dep1to0.size());
        satisfied1to0.resize(dep1to0.size(), false);
        for (const auto &d : dep1to0) {
            coeffSets1to0.push_back(
                intersect(coefficients(apply(d, legalityMap)),
                          coefficients(apply(d, boundingMap))));
            satSets1to0.push_back(coefficients(apply(d, legalityMap), 1));
        }
    }

    // construct the map from coefficients to optimize targets
    PBMapBuilder builder;

    // the coefficients set includes following dimensions:
    // 1. (nParams + 1) bounding coefficients
    auto cBounds = builder.newInputs(nParams + 1, "cb");
    // 2. (nParams + 1 + nestLevel0) loop 0 permuting coefficients
    auto c0Params = builder.newInputs(nParams + 1, "c0p");
    auto c0Iters = builder.newInputs(nestLevel0, "c0i");
    auto c0 = c0Params;
    c0.insert(c0.end(), c0Iters.begin(), c0Iters.end());
    // 3. (nParams + 1 + nestLevel1) loop 1 permuting coefficients
    auto c1Params = builder.newInputs(nParams + 1, "c1p");
    auto c1Iters = builder.newInputs(nestLevel1, "c1i");
    auto c1 = c1Params;
    c1.insert(c1.end(), c1Iters.begin(), c1Iters.end());

    // optimize targets
    // 1. the distance bounds go first, as main optimizing targets
    builder.addOutputs(cBounds);
    // 2.1. sum of coefficients absolute for loop 0
    auto c0Sum = builder.newOutput("c0sum");
    // 2.2. binary decision variable for avoiding zeros
    auto delta0 = builder.newOutput("d0");
    auto delta0L = builder.newOutput("d0l");
    // 2.3. reversed coefficients of loop 0 iterations
    //      they are reversed because we want to select outer loops
    //      earlier, preserving the original loop order
    builder.addOutputs(c0Iters | views::reverse);
    // 2.4. coefficients of loop 0 params and constant
    builder.addOutputs(c0Params);
    // 3.1. sum of coefficients absolute for loop 1
    auto c1Sum = builder.newOutput("c1sum");
    // 3.2. binary decision variable for avoiding zeros
    auto delta1 = builder.newOutput("d1");
    auto delta1L = builder.newOutput("d1l");
    // 3.3. reversed coefficients of loop 1 iterations
    builder.addOutputs(c1Iters | views::reverse);
    // 3.4. coefficients of loop 1 params and constant
    builder.addOutputs(c1Params);

    // the constraints on one loop
    builder.addConstraints({
        absSumConstraint(c0Sum, c0),
        nonZeroConstraint(c0Iters, delta0),
        delta0L >= 0,
        delta0L <= 1,
        absSumConstraint(c1Sum, c1),
        nonZeroConstraint(c1Iters, delta1),
        delta1L >= 0,
        delta1L <= 1,
    });

    auto optimizeMap = builder.build(ctx);
    auto revOptimizeMap = reverse(optimizeMap);

    PBSetBuilder orthoSetBuilder;
    orthoSetBuilder.addVars(builder.outputs());

    std::vector<std::vector<int>> c0ParamValue, c0IterValue;
    std::vector<std::vector<int>> c1ParamValue, c1IterValue;
    // start computing permuted dimensions
    int fusedLevel;
    for (fusedLevel = 0; fusedLevel < std::min(nestLevel0, nestLevel1);
         ++fusedLevel) {
        //! FIXME: handle parameters from loads
        auto problem = universeSet(
            spaceSetAlloc(ctx, 0, (nParams + 1) * 3 + nestLevel0 + nestLevel1));
        // constructing the coefficients' space
        for (const auto &[satisfied, coeffSet] :
             views::zip(views::concat(satisfied0, satisfied1, satisfied1to0),
                        views::concat(coeffSets0, coeffSets1, coeffSets1to0)))
            if (!satisfied)
                problem = intersect(std::move(problem), coeffSet);

        // construct orthogonal constraints
        auto orthoConstraint = [&](const auto &cIterValue, const auto &cIters,
                                   const auto &deltaL) {
            auto ortho = orthogonalMatrix(cIterValue);
            std::vector<PBBuildExpr> orthoDots;
            orthoDots.reserve(ortho.size());
            for (auto &&o : ortho) {
                orthoDots.emplace_back(0);
                ASSERT(o.size() == cIters.size());
                for (size_t i = 0; i < cIters.size(); ++i)
                    orthoDots.back() += o[i] * cIters[i];
            }
            return nonZeroConstraint(orthoDots, deltaL);
        };
        if (fusedLevel > 0) {
            orthoSetBuilder.addConstraint(
                orthoConstraint(c0IterValue, c0Iters, delta0L));
            orthoSetBuilder.addConstraint(
                orthoConstraint(c1IterValue, c1Iters, delta1L));
        }
        auto orthoSet = orthoSetBuilder.build(ctx);
        orthoSetBuilder.clearConstraints();

        // map the coefficients to optimize targets, and perform optimization
        auto solution =
            lexmin(intersect(std::move(orthoSet), apply(problem, optimizeMap)));
        if (solution.empty())
            break;
        std::cout << solution << std::endl;

        auto solutionVals =
            sample(apply(std::move(solution), revOptimizeMap)).coordinates();
        auto optimized = solutionVals | views::transform([&](const PBVal &val) {
                             ASSERT(val.denSi() == 1);
                             return val.numSi();
                         }) |
                         ranges::to<std::vector>();

        // check and exclude fake fusion
        auto loopSetToRange = [&](const PBSet &loopSet, int coeffBase,
                                  int nestLevel) {
            PBMapBuilder builder;
            auto p = builder.newInputs(nParams, "p");
            auto x = builder.newInputs(nestLevel, "x");
            PBBuildExpr result = 0;
            for (int i = 0; i < nParams; ++i)
                result += optimized[coeffBase + i] * p[i];
            result += optimized[coeffBase + nParams];
            for (int i = 0; i < nestLevel; ++i)
                result += optimized[coeffBase + nParams + 1 + i] * x[i];
            builder.addOutput(result);
            return apply(loopSet, builder.build(ctx));
        };
        auto loop0Range = loopSetToRange(loop0Set, nParams + 1, nestLevel0);
        PBSet loop1Range = loopSetToRange(
            loop1Set, (nParams + 1) * 2 + nestLevel0, nestLevel1);
        // if the two loops has no overlap on the result axis, they are not
        // actually fused so we bail out
        if (intersect(
                // range of values that less than the maximum of loop 0
                apply(lexmax(loop0Range), lexGT(spaceSetAlloc(ctx, 0, 1))),
                // ... and from loop 1
                loop1Range)
                // ... don't overlap
                .empty())
            break;

        // save coefficients' values
        c0ParamValue.push_back({
            optimized.begin() + (nParams + 1),
            optimized.begin() + (nParams + 1) * 2,
        });
        c0IterValue.push_back({
            optimized.begin() + (nParams + 1) * 2,
            optimized.begin() + (nParams + 1) * 2 + nestLevel0,
        });
        c1ParamValue.push_back({
            optimized.begin() + (nParams + 1) * 2 + nestLevel0,
            optimized.begin() + (nParams + 1) * 3 + nestLevel0,
        });
        c1IterValue.push_back({
            optimized.begin() + (nParams + 1) * 3 + nestLevel0,
            optimized.end(),
        });
    }

    ASSERT(c0ParamValue.size() == unsigned(fusedLevel));
    ASSERT(c0IterValue.size() == unsigned(fusedLevel));
    ASSERT(c1ParamValue.size() == unsigned(fusedLevel));
    ASSERT(c1IterValue.size() == unsigned(fusedLevel));

    // fill rest dimensions
    auto restOrtho0 = orthogonalMatrix(c0IterValue);
    c0IterValue.insert(c0IterValue.end(), restOrtho0.begin(), restOrtho0.end());
    for (int i = 0; i < nestLevel0 - fusedLevel; ++i)
        c0ParamValue.emplace_back(nParams + 1, 0);

    auto restOrtho1 = orthogonalMatrix(c1IterValue);
    c1IterValue.insert(c1IterValue.end(), restOrtho1.begin(), restOrtho1.end());
    for (int i = 0; i < nestLevel1 - fusedLevel; ++i)
        c1ParamValue.emplace_back(nParams + 1, 0);

    std::cout << "c0 =" << std::endl;
    std::cout << "[ ";
    for (int i = 0; i < nestLevel0; ++i) {
        if (i != 0)
            std::cout << "  ";
        std::cout << "[ " << c0ParamValue[i] << " | " << c0IterValue[i] << " ]";
        if (i != nestLevel0)
            std::cout << "," << std::endl;
        else
            std::cout << " ]" << std::endl;
    }

    std::cout << "c1 =" << std::endl;
    std::cout << "[ ";
    for (int i = 0; i < nestLevel1; ++i) {
        if (i != 0)
            std::cout << "  ";
        std::cout << "[ " << c1ParamValue[i] << " | " << c1IterValue[i] << " ]";
        if (i != nestLevel1)
            std::cout << "," << std::endl;
        else
            std::cout << " ]" << std::endl;
    }

    //! TODO: compute permutation according to the coefficients via
    //! reverse(PBMap)
    std::vector<Var> fusedLoopsVar, remainLoop0Var, remainLoop1Var;
    for (int i = 0; i < fusedLevel; ++i)
        fusedLoopsVar.push_back(makeVar("fuse_i" + toString(i)));
    for (int i = fusedLevel; i < nestLevel0; ++i)
        remainLoop0Var.push_back(makeVar("rem0_i" + toString(i)));
    for (int i = fusedLevel; i < nestLevel1; ++i)
        remainLoop1Var.push_back(makeVar("rem1_i" + toString(i)));
    std::vector<Expr> loop0VarReplacement, loop1VarReplacement;
    {}

    //! TODO: transform original ast according to computed permutation

    return {original_ast, {}};
}

} // namespace freetensor
