#include <cmath>
#include <isl/set.h>
#include <numeric>
#include <string>
#include <vector>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <expr.h>
#include <math/parse_pb_expr.h>
#include <math/presburger.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/normalize_loops.h>
#include <pass/pb_simplify.h>
#include <pass/shrink_for.h>
#include <pass/sink_var.h>
#include <schedule.h>
#include <schedule/fuse.h>
#include <schedule/pluto.h>
#include <serialize/load_ast.h>
#include <serialize/mangle.h>

namespace freetensor {

namespace {

PBBuildExpr nonZeroConstraint(const std::vector<PBBuildExpr> &vars,
                              const PBBuildExpr &delta, int varBound = 8) {
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

PBBuildExpr ctlValConstraint(PBBuildExpr c, PBBuildExpr v, PBBuildExpr ctl,
                             int varBound = 8) {
    // rule:
    // ctl = 0 -> c = 0 and 1 <= v <= varBound
    // ctl = 1 -> c = v and 1 <= v <= varBound
    return c - v - varBound * ctl + varBound >= 0 && c - v - ctl + 1 <= 0 &&
           -c + varBound * ctl >= 0 && -c + ctl <= 0;
}

PBBuildExpr absConstraint(PBBuildExpr x, PBBuildExpr abs, PBBuildExpr ctl,
                          int varBound = 8) {
    // rule:
    // ctl = 0 -> abs = x and 0 <= x <= varBound
    // ctl = 1 -> abs = -x and -varBound <= x <= 1
    return x + abs + 2 * varBound * ctl - 2 * varBound <= 0 && x + abs >= 0 &&
           x - abs + 2 * varBound * ctl >= 0 && x - abs + 2 * ctl <= 0;
}

std::vector<std::vector<int>>
orthogonalMatrix(const std::vector<std::vector<int>> &vectors) {
    // sanity check
    ASSERT(vectors.size() > 0);
    for (size_t i = 1; i < vectors.size(); ++i)
        ASSERT(vectors[i].size() == vectors[0].size());
    int nDims = vectors[0].size();

    PBSetBuilder builder;
    auto abss = builder.newVars(nDims, "abs_x");
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
    builder.addConstraints(views::zip_with(
        [](auto &&abs, auto &&x) { return abs >= x && abs >= -x; }, abss,
        vars));
    builder.addConstraints(vectors | views::transform(fOrtho));
    PBSet orthogonalSet = builder.build(ctx);
    builder.clearConstraints();

    std::vector<std::vector<int>> result;
    while (!orthogonalSet.empty()) {
        // lexmin and sample to solve
        auto solution = sample(lexmin(orthogonalSet)).coordinates();
        // extract the vector into result
        std::vector<int> v;
        v.reserve(nDims);
        for (int i = nDims; i < 2 * nDims; ++i) {
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
    ID outer_, target0_, target1_;

  public:
    InjectFakeAccess(const ID &outer, const ID &target0, const ID &target1)
        : outer_(outer), target0_(target0), target1_(target1) {}

  protected:
    Stmt visit(const For &op) override {
        if (op->id() != target0_ && op->id() != target1_)
            return Mutator::visit(op);

        auto newBody = makeStmtSeq({
            makeStore(FAKE_ACCESS_VAR, {}, makeIntConst(0)),
            op->body_,
        });

        return makeFor(op->iter_, op->begin_, op->end_, op->step_, op->len_,
                       op->property_, newBody, op->metadata(), op->id());
    }

    Stmt visitStmt(const Stmt &op) override {
        if (op->id() != outer_)
            return Mutator::visitStmt(op);

        return makeVarDef(FAKE_ACCESS_VAR,
                          makeBuffer(makeTensor({}, DataType::Int32),
                                     AccessType::Cache, MemType::ByValue),
                          std::nullopt, Mutator::visitStmt(op), false);
    }
};

PBMap combineExternal(PBMap l2e, PBCtx &ctx) {
    auto nParams = l2e.nParamDims();
    // combined <-> first met original
    std::map<std::string, std::string> orig2comb;
    std::map<std::string, PBBuildExpr> comb2origVar;
    PBSetBuilder builder;
    for (int i = 0; i < nParams; ++i) {
        std::string orig = isl_map_get_dim_name(l2e.get(), isl_dim_param, i);
        auto origVar = builder.newVar(orig);
        std::string comb;
        if (auto pos = orig.find("earlier"); pos != std::string::npos)
            comb = orig.substr(0, pos);
        else if (auto pos = orig.find("later"); pos != std::string::npos)
            comb = orig.substr(0, pos);
        else
            ASSERT(false && "FindDeps returns unexpected parameter name");

        if (!comb2origVar.contains(comb)) {
            comb2origVar[comb] = origVar;
            orig2comb[orig] = comb;
        } else
            builder.addConstraint(comb2origVar[comb] == origVar);
    }
    auto eqConstraints = builder.build(ctx);
    eqConstraints = isl_set_move_dims(eqConstraints.move(), isl_dim_param, 0,
                                      isl_dim_set, 0, nParams);
    if (l2e !=
        PBMap(isl_map_intersect_params(l2e.copy(), eqConstraints.move())))
        throw InvalidSchedule("PlutoFuse: External parameter(s) should not "
                              "change between during the fused loops");

    for (int i = nParams - 1; i >= 0; --i) {
        std::string orig = isl_map_get_dim_name(l2e.get(), isl_dim_param, i);
        if (orig2comb.contains(orig))
            l2e = isl_map_set_dim_name(l2e.move(), isl_dim_param, i,
                                       orig2comb[orig].c_str());
        else
            l2e = isl_map_remove_dims(l2e.move(), isl_dim_param, i, 1);
    }

    return l2e;
}

std::pair<std::vector<IterAxis>, std::vector<IterAxis>>
extractVarAxes(const std::vector<IterAxis> &axes, const For &targetLoop) {
    bool inLoop = false;
    std::vector<IterAxis> outer, inner;
    for (auto &&axis : axes)
        if (axis.iter_->nodeType() == ASTNodeType::Var) {
            if (axis.iter_.as<VarNode>()->name_ == targetLoop->iter_)
                inLoop = true;
            (inLoop ? inner : outer).emplace_back(axis);
        }
    if (!inLoop)
        ERROR("PlutoFuse: target loop not found in fake access");
    return {std::move(outer), std::move(inner)};
}

std::pair<int, std::vector<int>> findIterFromAP(const AccessPoint &ap,
                                                const std::string var) {
    std::vector<int> outerDims;
    int n = ap.iter_.size();
    outerDims.reserve(n);
    for (int i = 0; i < n; ++i)
        if (ap.iter_[i].iter_->nodeType() == ASTNodeType::Var) {
            if (ap.iter_[i].iter_.as<VarNode>()->name_ == var)
                return std::pair{i, std::move(outerDims)};
            else
                outerDims.push_back(i);
        }
    ASSERT(false);
}

struct ReplaceVar : public Mutator {
    std::unordered_map<std::string, Expr> replaceMap_;

    Expr visit(const Var &op) override {
        if (replaceMap_.contains(op->name_))
            return replaceMap_.at(op->name_);
        else
            return Mutator::visit(op);
    }
};

struct PermuteInfo {
    std::vector<Expr> vars_;
    Expr cond_;

    PermuteInfo(const std::vector<std::vector<int>> &cParamValue,
                const std::vector<std::vector<int>> &cIterValue,
                const std::vector<Expr> &paramExprs,
                const std::vector<IterAxis> &oldLoopAxes,
                const std::vector<std::string> &loopVars, const PBCtx &ctx,
                const PBSet &loopSet) {
        size_t nParams = paramExprs.size(), nestLevel = loopVars.size();
        ASSERT(oldLoopAxes.size() == loopVars.size());

        PBMapBuilder builder;
        auto params = builder.newOutputs(nParams);
        auto newIter = builder.newInputs(nestLevel);
        auto oldIter = builder.newOutputs(nestLevel);
        for (size_t i = 0; i < nestLevel; ++i) {
            ASSERT(nParams + 1 == cParamValue[i].size());
            PBBuildExpr iter = 0;
            for (size_t j = 0; j < nParams; ++j)
                iter += cParamValue[i][j] * params[j];
            iter += cParamValue[i][nParams];
            for (size_t j = 0; j < nestLevel; ++j)
                iter += cIterValue[i][j] * oldIter[j];
            builder.addConstraint(iter == newIter[i]);
        }
        auto newToOld = builder.build(ctx);
        newToOld = intersectRange(std::move(newToOld), loopSet);
        newToOld = moveDimsOutputToInput(std::move(newToOld), 0, nParams, 0);

        auto func = parseSimplePBFunc(toString(PBFunc(newToOld)));
        ASSERT(func.args_.size() == unsigned(nParams + nestLevel));
        ASSERT(func.values_.size() == unsigned(nestLevel));

        ReplaceVar renamer;
        for (size_t i = 0; i < nParams; ++i)
            renamer.replaceMap_[func.args_[i]] = paramExprs[i];
        for (size_t i = 0; i < nestLevel; ++i)
            renamer.replaceMap_[func.args_[nParams + i]] = makeVar(loopVars[i]);

        vars_.reserve(nestLevel);
        for (size_t i = 0; i < nestLevel; ++i) {
            auto rawIter = renamer(func.values_[i]);
            if (oldLoopAxes[i].negStep_)
                vars_.push_back(makeSub(makeIntConst(0), rawIter));
            else
                vars_.push_back(rawIter);
        }
        cond_ = renamer(func.cond_);
    }
};

struct PlutoFuse : public Mutator {
    // original loops ID
    ID loop0Id_, loop1Id_;
    // new var names
    const std::vector<std::string> &fusedLoopsVar_, &remainLoop0Var_,
        &remainLoop1Var_;
    // permute info: what to replace the original vars and extra conditions
    const PermuteInfo &permute0_, &permute1_;

    For loop0_ = nullptr, loop1_ = nullptr;
    ID fusedId_;

    PlutoFuse(const ID &loop0Id, const ID &loop1Id,
              const std::vector<std::string> &fusedLoopsVar,
              const std::vector<std::string> &remainLoop0Var,
              const std::vector<std::string> &remainLoop1Var,
              const PermuteInfo &permute0, const PermuteInfo &permute1)
        : loop0Id_(loop0Id), loop1Id_(loop1Id), fusedLoopsVar_(fusedLoopsVar),
          remainLoop0Var_(remainLoop0Var), remainLoop1Var_(remainLoop1Var),
          permute0_(permute0), permute1_(permute1) {}

    Stmt visit(const For &op) override {
        if (op->id() == loop0Id_) {
            loop0_ = op;
            return makeStmtSeq({});
        } else if (op->id() != loop1Id_)
            return Mutator::visit(op);

        // At loop 1, replace with the fused loop
        loop1_ = op;

        // sanity check: loop 0 should be already found
        ASSERT(loop0_.isValid());
        // sanity check: vars size match
        ASSERT(fusedLoopsVar_.size() + remainLoop0Var_.size() ==
               permute0_.vars_.size());
        ASSERT(fusedLoopsVar_.size() + remainLoop1Var_.size() ==
               permute1_.vars_.size());

        // sanity check and construct the vars replacing map
        ReplaceVar loop0Replace, loop1Replace;
        Stmt loop0Inner = loop0_, loop1Inner = loop1_;
        for (auto &&replaced : permute0_.vars_) {
            ASSERT(loop0Inner->nodeType() == ASTNodeType::For);
            loop0Replace.replaceMap_[loop0Inner.as<ForNode>()->iter_] =
                replaced;
            loop0Inner = loop0Inner.as<ForNode>()->body_;
        }
        for (auto &&replaced : permute1_.vars_) {
            ASSERT(loop1Inner->nodeType() == ASTNodeType::For);
            loop1Replace.replaceMap_[loop1Inner.as<ForNode>()->iter_] =
                replaced;
            loop1Inner = loop1Inner.as<ForNode>()->body_;
        }

        auto genInner = [&](const PermuteInfo &pi, const For &loop,
                            const Stmt &loopInner, ReplaceVar replacer,
                            const std::vector<std::string> &remainLoopVar) {
            auto body = replacer(loopInner);
            body = makeIf(pi.cond_, body);
            for (int i = remainLoopVar.size() - 1; i >= 0; --i)
                body = makeFor(
                    remainLoopVar[i], makeIntConst(INT32_MIN),
                    makeIntConst(INT32_MAX), makeIntConst(1),
                    makeIntConst(int64_t(INT32_MAX) - INT32_MIN),
                    Ref<ForProperty>::make(), body,
                    makeMetadata("pluto_fuse.inner." + toString(i), loop));
            return body;
        };
        auto fusedLoop = makeStmtSeq({
            genInner(permute0_, loop0_, loop0Inner, loop0Replace,
                     remainLoop0Var_),
            genInner(permute1_, loop1_, loop1Inner, loop1Replace,
                     remainLoop1Var_),
        });
        for (int i = fusedLoopsVar_.size() - 1; i >= 0; --i)
            fusedLoop = makeFor(fusedLoopsVar_[i], makeIntConst(INT32_MIN),
                                makeIntConst(INT32_MAX), makeIntConst(1),
                                makeIntConst(int64_t(INT32_MAX) - INT32_MIN),
                                Ref<ForProperty>::make(), fusedLoop,
                                makeMetadata("pluto_fuse.fused." + toString(i),
                                             loop0_, loop1_));
        fusedId_ = fusedLoop->id();
        return fusedLoop;
    }
};

std::pair<Stmt, std::pair<ID, int>>
plutoFuseImpl(Stmt ast, const ID &loop0Id, const ID &loop1Id, int _nestLevel0,
              int _nestLevel1, int fusableOverlapThreshold,
              int fusableNonOverlapTolerance, bool doSimplify) {
    bool hoisted = false;
    // try with vardef hoisted only if they are not at the same level
    if (findStmt(ast, loop0Id)->parent() != findStmt(ast, loop1Id)->parent()) {
        hoisted = true;
        ast = hoistVarOverStmtSeq(ast);
    }

    // check accessed vardefs: those vardefs accessed by loop1 should not have
    // their shapes modified in loop0
    CheckFuseAccessible check(loop0Id, loop1Id);
    check.check(ast);

    // count maximum count of perfectly nested loops at loop0 and loop1
    auto countPerfectNest = [](const For &loop, int n) {
        Stmt inner;
        if (n == 0) {
            for (inner = loop; inner->nodeType() == ASTNodeType::For;
                 inner = inner.as<ForNode>()->body_)
                n++;
        } else {
            int expectedN = n;
            n = 0;
            for (inner = loop;
                 inner->nodeType() == ASTNodeType::For && n < expectedN;
                 inner = inner.as<ForNode>()->body_)
                n++;
            if (n < expectedN)
                throw InvalidSchedule(
                    "PlutoFuse: not enough loop nests found for " +
                    toString(loop));
        }
        return std::pair{n, inner->parentStmt()->id()};
    };
    auto [nestLevel0, inner0] =
        countPerfectNest(check.loop0().loop_, _nestLevel0);
    auto [nestLevel1, inner1] =
        countPerfectNest(check.loop1().loop_, _nestLevel1);

    // inject fake accesses to extract loop space
    auto fakeAccessAst =
        InjectFakeAccess(check.loop0().loop_->parent().as<StmtNode>()->id(),
                         inner0, inner1)(ast);

    auto loop0 = findStmt(fakeAccessAst, loop0Id).as<ForNode>();
    auto loop1 = findStmt(fakeAccessAst, loop1Id).as<ForNode>();

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
    std::string loop0SetStr, loop1SetStr;
    std::vector<IterAxis> outerAxes, loop0Axes, loop1Axes;

    auto getDeps = [&](const For &l0, int n0, const For &l1, int n1,
                       bool handleFakeAccess = false) mutable {
        std::unordered_set<std::string> deps;
        std::mutex m;

        FindDeps()
            .noProjectOutPrivateAxis(true)
            .ignoreReductionWAW(false)
            .filterEarlier([&](const AccessPoint &p) {
                return p.stmt_->ancestorById(l0->id()).isValid();
            })
            .filterLater([&](const AccessPoint &p) {
                return p.stmt_->ancestorById(l1->id()).isValid();
            })
            .direction(outersSame)(
                fakeAccessAst, unsyncFunc([&](const Dependence &d) {
                    auto hMap = d.later2EarlierIter_;

                    // We reuse the same transformation for fake access.
                    // However, the final later2earlier drops all non-nearest
                    // and lacks information about the full loop sets. As such,
                    // we use the AllPossible one for it.
                    if (d.var_ == FAKE_ACCESS_VAR) {
                        // only process fake access when specified
                        if (handleFakeAccess) {
                            hMap = d.later2EarlierIterAllPossible_;
                            std::tie(outerAxes, loop0Axes) =
                                extractVarAxes(d.earlier_.iter_, loop0);
                            loop1Axes =
                                extractVarAxes(d.later_.iter_, loop1).second;
                        } else
                            return;
                    }

                    // combine external params from earlier and later, since we
                    // don't expect them to change during the two loops in Pluto
                    hMap = combineExternal(std::move(hMap), d.presburger_);

                    auto nRealParams = hMap.nParamDims();

                    // remove inner dims for earlier
                    auto [pos0, outerDims0] =
                        findIterFromAP(d.earlier_, l0->iter_);
                    pos0 += n0;
                    hMap = projectOutOutputDims(std::move(hMap), pos0,
                                                hMap.nOutDims() - pos0);
                    pos0 -= n0;
                    // drop all constant dimensions
                    for (int i = outerDims0.size() - 1; i >= 0;
                         pos0 = outerDims0[i--])
                        hMap = projectOutOutputDims(std::move(hMap),
                                                    outerDims0[i] + 1,
                                                    pos0 - outerDims0[i] - 1);
                    hMap = projectOutOutputDims(std::move(hMap), 0, pos0);
                    // move outer dimensions to params
                    hMap = moveDimsOutputToParam(
                        std::move(hMap), 0, outerDims0.size(), nRealParams);
                    for (size_t i = 0; i < outerDims0.size(); ++i)
                        hMap = isl_map_set_dim_name(
                            hMap.move(), isl_dim_param, nRealParams + i,
                            ("out_" + std::to_string(i)).c_str());

                    // remove inner dims for later
                    auto [pos1, outerDims1] =
                        findIterFromAP(d.later_, l1->iter_);
                    pos1 += n1;
                    hMap = projectOutInputDims(std::move(hMap), pos1,
                                               hMap.nInDims() - pos1);
                    pos1 -= n1;
                    // simply drop all outers since we already specified
                    // outersSame
                    hMap = projectOutInputDims(std::move(hMap), 0, pos1);

                    // if fake access, set the loop sets instead of store deps
                    if (d.var_ == FAKE_ACCESS_VAR) {
                        // hMap is later -> earlier, hence loop1 -> loop0.
                        loop1SetStr = toString(std::move(domain(hMap)));
                        loop0SetStr = toString(std::move(range(hMap)));
                        return;
                    }

                    // flatten to set for later coefficients computation;
                    // later dimensions first, so the first half would be
                    // target, and second half being source
                    auto hSet = flattenMapToSet(std::move(hMap));
                    // overapproximate to allow coefficients on strided
                    // dependences
                    hSet = isl_set_remove_unknown_divs(hSet.move());
                    auto strSet = toString(std::move(hSet));

                    std::lock_guard l(m);
                    // do some deduplicate (deps is a set)
                    deps.insert(std::move(strSet));
                }));
        std::vector<PBSet> depsVec;
        for (auto &&d : deps)
            depsVec.emplace_back(ctx, d);
        return depsVec;
    };

    // find dependences
    auto dep0 = getDeps(loop0, nestLevel0, loop0, nestLevel0);
    auto dep1 = getDeps(loop1, nestLevel1, loop1, nestLevel1);
    auto dep1to0 = getDeps(loop0, nestLevel0, loop1, nestLevel1, true);

    // construct loop sets
    PBSet loop0Set(ctx, loop0SetStr), loop1Set(ctx, loop1SetStr);

    // align external params and move to set dimensions
    const auto [nParams, paramExprs] = [&] {
        auto paramsSpace = spaceSetAlloc(ctx, 0, 0);
        for (auto &d : {&dep0, &dep1, &dep1to0})
            for (const auto &dd : *d)
                paramsSpace = isl_space_align_params(paramsSpace.move(),
                                                     PBSpace(dd).move());
        auto n = isl_space_dim(paramsSpace.get(), isl_dim_param);

        auto align = [&](PBSet &s) {
            s = isl_set_move_dims(
                isl_set_align_params(s.move(), paramsSpace.copy()), isl_dim_set,
                0, isl_dim_param, 0, n);
        };

        for (auto &d : {&dep0, &dep1, &dep1to0})
            for (auto &dd : *d)
                align(dd);
        align(loop0Set);
        align(loop1Set);

        std::vector<Expr> params;
        params.reserve(n);
        for (size_t i = 0; i < n - outerAxes.size(); ++i) {
            std::string name =
                isl_space_get_dim_name(paramsSpace.get(), isl_dim_param, i);
            auto pos = name.find("__ext__");
            ASSERT(pos != std::string::npos);
            params.push_back(
                loadAST(unmangle(name.substr(0, pos))).as<ExprNode>());
        }
        for (size_t i = 0; i < outerAxes.size(); ++i)
            params.push_back(outerAxes[i].iter_);

        return std::pair{n, params};
    }();

    // constraints for bounding and valid coefficients
    std::vector<PBSet> coeffSets0, coeffSets1, coeffSets1to0;
    // set of coefficients satisifying the dependence
    // though lower bounds are presented in optimizedMap input, we still have to
    // compute satisfication status seperately since the lb variables are not in
    // the optimize targets for better performance.
    std::vector<PBSet> satSets0, satSets1, satSets1to0;
    // whether a dependence have been satisfied already
    std::vector<bool> satisfied0, satisfied1, satisfied1to0;

    // process dependences inside loop 0
    if (!dep0.empty()) {
        // ti (target iter) and si (source iter) both on loop 0.
        // this part has no constraint on c1.
        PBMapBuilder builder;
        auto p = builder.newInputs(nParams, "p");
        auto ti = builder.newInputs(nestLevel0, "ti");
        auto si = builder.newInputs(nestLevel0, "si");

        // legality problem:
        // c0 * ti - c0 * si - lb >= 0
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
        // bounds coefficients applied to params
        builder.addOutputs(p);
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
        for (auto &&[i, d] : views::enumerate(dep0)) {
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
        auto p = builder.newInputs(nParams, "p");
        auto ti = builder.newInputs(nestLevel1, "ti");
        auto si = builder.newInputs(nestLevel1, "si");

        // legality problem:
        // c1 * ti - c1 * si - lb >= 0
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
        // bounds coefficients applied to params
        builder.addOutputs(p);
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
        for (auto &&[i, d] : views::enumerate(dep1)) {
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
        // later first, so p0 and i0 goes first
        PBMapBuilder builder;
        auto p = builder.newInputs(nParams, "p");
        auto i1 = builder.newInputs(nestLevel1, "i1");
        auto i0 = builder.newInputs(nestLevel0, "i0");

        auto negate = views::transform([](auto &&x) { return -x; });

        // legality problem:
        // c1 * i1 - c0 * i0 - lb >= 0
        // bounds coefficients, no effect
        builder.addOutputs(views::repeat_n(0, nParams + 1));
        // c0
        builder.addOutputs(p | negate);
        builder.addOutput(-1);
        builder.addOutputs(i0 | negate);
        // c1
        builder.addOutputs(p);
        builder.addOutput(1);
        builder.addOutputs(i1);
        auto legalityMap = builder.build(ctx);
        builder.clearOutputs();

        // bounding problem:
        // c * parameters - (c1 * i1 - c0 * i0) >= 0
        // bounds coefficients
        builder.addOutputs(p);
        builder.addOutput(1);
        // c0
        builder.addOutputs(p);
        builder.addOutput(1);
        builder.addOutputs(i0);
        // c1
        builder.addOutputs(p | negate);
        builder.addOutput(-1);
        builder.addOutputs(i1 | negate);
        auto boundingMap = builder.build(ctx);

        coeffSets1to0.reserve(dep1to0.size());
        satSets1to0.reserve(dep1to0.size());
        satisfied1to0.resize(dep1to0.size(), false);
        for (auto &&[i, d] : views::enumerate(dep1to0)) {
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
    // 3. (nParams + 1 + nestLevel1) loop 1 permuting coefficients
    auto c1Params = builder.newInputs(nParams + 1, "c1p");
    auto c1Iters = builder.newInputs(nestLevel1, "c1i");

    // optimize targets
    // 1. Pluto+ targets
    //    for each bounding coefficients except the last one (for constant),
    //    we need to minimize its absolute value to avoid -inf results
    for (int i = 0; i < nParams; ++i) {
        auto abs = builder.newOutput("abs_cb" + toString(i));
        builder.addConstraint(abs >= cBounds[i] && abs >= -cBounds[i]);
        builder.addOutput(cBounds[i]);
    }
    builder.addOutput(cBounds[nParams]);
    // 2.1. reversed coefficients of loop 0 iterations
    //      they are reversed because we want to select outer loops earlier,
    //      preserving the original loop order; also, only one axis should
    //      involve (prevent any skewness!)
    // the shared abstract value for all loop 0 iter coefficients
    auto c0IterVal = builder.newOutput("c0i_val");
    PBBuildExpr sumNnzCtl = 0;
    std::vector<PBBuildExpr> nnzCtl0(nestLevel0, PBBuildExpr());
    for (int i = nestLevel0 - 1; i >= 0; --i) {
        nnzCtl0[i] = builder.newOutput("nnz_ctl_c0i" + toString(i));
        auto negCtl = builder.newOutput("neg_ctl_c0i" + toString(i));
        auto abs = builder.newOutput("abs_c0i" + toString(i));
        builder.addOutput(c0Iters[i]);
        builder.addConstraint(ctlValConstraint(abs, c0IterVal, nnzCtl0[i]));
        builder.addConstraint(absConstraint(c0Iters[i], abs, negCtl));
        sumNnzCtl += nnzCtl0[i];
    }
    builder.addConstraint(sumNnzCtl == 1);
    // 2.2. coefficients of loop 0 params and constant
    for (int i = 0; i < nParams + 1; ++i) {
        auto abs = builder.newOutput("abs_c0p" + toString(i));
        builder.addConstraint(abs >= c0Params[i] && abs >= -c0Params[i]);
        builder.addOutput(c0Params[i]);
    }
    // 3.1. reversed coefficients of loop 1 iterations
    auto c1IterVal = builder.newOutput("c1i_val");
    sumNnzCtl = 0;
    std::vector<PBBuildExpr> nnzCtl1(nestLevel1, PBBuildExpr());
    for (int i = nestLevel1 - 1; i >= 0; --i) {
        nnzCtl1[i] = builder.newOutput("nnz_ctl_c1i" + toString(i));
        auto negCtl = builder.newOutput("neg_ctl_c1i" + toString(i));
        auto abs = builder.newOutput("abs_c1i" + toString(i));
        builder.addOutput(c1Iters[i]);
        builder.addConstraint(ctlValConstraint(abs, c1IterVal, nnzCtl1[i]));
        builder.addConstraint(absConstraint(c1Iters[i], abs, negCtl));
        sumNnzCtl += nnzCtl1[i];
    }
    builder.addConstraint(sumNnzCtl == 1);
    // 3.2. coefficients of loop 1 params and constant
    for (int i = 0; i < nParams + 1; ++i) {
        auto abs = builder.newOutput("abs_c1p" + toString(i));
        builder.addConstraint(abs >= c1Params[i] && abs >= -c1Params[i]);
        builder.addOutput(c1Params[i]);
    }

    auto optimizeMap = builder.build(ctx);
    auto revOptimizeMap = reverse(optimizeMap);

    // constrain the bounding function to be >= 0, avoiding unbound optimum when
    // no dependence occurs
    {
        PBMapBuilder builder;

        auto params = builder.newInputs(nParams, "p");
        auto iters0 = builder.newInputs(nestLevel0, "i0_");

        builder.addOutputs(params);
        builder.addOutput(1);
        builder.addOutputs(
            views::repeat_n(0, (nParams + 1) * 2 + nestLevel0 + nestLevel1));
        optimizeMap =
            intersectDomain(std::move(optimizeMap),
                            coefficients(apply(loop0Set, builder.build(ctx))));
    }

    PBSetBuilder orthoSetBuilder;
    orthoSetBuilder.addVars(builder.outputs());

    std::vector<std::vector<int>> c0ParamValue, c0IterValue;
    std::vector<std::vector<int>> c1ParamValue, c1IterValue;
    // start computing permuted dimensions
    int fusedLevel, parallelCount = 0;
    int hugeNonOverlapFusedLevel = -1, hugeNonOverlapCount = 0;
    bool isParallel = true;
    for (fusedLevel = 0; fusedLevel < std::min(nestLevel0, nestLevel1);
         ++fusedLevel) {
        //! FIXME: handle parameters from loads
        PBSet problem;
        // constructing the coefficients' space
        for (const auto &[satisfied, coeffSet] :
             views::zip(views::concat(satisfied0, satisfied1, satisfied1to0),
                        views::concat(coeffSets0, coeffSets1, coeffSets1to0)))
            if (!satisfied) {
                if (!problem.isValid())
                    problem = coeffSet;
                else
                    problem = intersect(std::move(problem), coeffSet);
            }

        if (!problem.isValid())
            problem = universeSet(spaceSetAlloc(
                ctx, 0, (nParams + 1) * 3 + nestLevel0 + nestLevel1));

        // construct orthogonal constraints
        auto orthoConstraint = [&](const auto &cIterValue, const auto &nnzCtl) {
            PBBuildExpr ret = true;
            for (auto &&cIterAxis : cIterValue) {
                for (auto &&[i, val] : views::enumerate(cIterAxis))
                    if (val != 0) {
                        ret = ret && nnzCtl[i] == 0;
                        break;
                    }
            }
            return ret;
        };
        if (fusedLevel > 0) {
            orthoSetBuilder.addConstraint(
                orthoConstraint(c0IterValue, nnzCtl0));
            orthoSetBuilder.addConstraint(
                orthoConstraint(c1IterValue, nnzCtl1));
        }
        auto orthoSet = orthoSetBuilder.build(ctx);
        orthoSetBuilder.clearConstraints();

        // map the coefficients to optimize targets, and perform optimization
        auto solution = apply(
            lexmin(intersect(std::move(orthoSet), apply(problem, optimizeMap))),
            revOptimizeMap);
        if (solution.empty())
            break;

        // check satisfied and mark; already satisfied dependences won't be
        // included in inner levels
        for (size_t i = 0; i < dep0.size(); ++i)
            if (!intersect(solution, satSets0[i]).empty())
                satisfied0[i] = true;
        for (size_t i = 0; i < dep1.size(); ++i)
            if (!intersect(solution, satSets1[i]).empty())
                satisfied1[i] = true;
        for (size_t i = 0; i < dep1to0.size(); ++i)
            if (!intersect(solution, satSets1to0[i]).empty())
                satisfied1to0[i] = true;

        auto solutionVals = sample(std::move(solution)).coordinates();
        auto optimized = solutionVals | views::transform([&](const PBVal &val) {
                             ASSERT(val.denSi() == 1);
                             return val.numSi();
                         }) |
                         ranges::to<std::vector>();

        // don't reset since we only count outer parallelism
        for (int i = 0; i < nParams + 1; ++i)
            isParallel = isParallel && optimized[i] == 0;
        if (isParallel)
            parallelCount++;

        // check and exclude fake fusion
        auto loopSetToRange = [&, nParams = nParams](const PBSet &loopSet,
                                                     int coeffBase,
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
            builder.addOutputs(p);
            builder.addOutput(result);
            auto projectedLoopRange = apply(loopSet, builder.build(ctx));
            return PBSet(isl_set_remove_divs(projectedLoopRange.move()));
        };
        auto loop0Range = loopSetToRange(loop0Set, nParams + 1, nestLevel0);
        auto loop1Range = loopSetToRange(
            loop1Set, (nParams + 1) * 2 + nestLevel0, nestLevel1);

        // overlap check 1: loop 0 & 1 should actually overlap
        {
            auto overlap = fixDim(universeSet(PBSpace(loop1Range)), nParams,
                                  fusableOverlapThreshold);
            // if the two loops has no overlap (for any possible parameters) on
            // the result axis, they are not actually fused so we bail out
            // "Exist P, R0 and R1 non-empty but R0 intersect R1 empty"
            auto R0 = loop0Range, R1 = sum(loop1Range, overlap);
            auto R01 = intersect(R0, R1);

            // project to only params
            R0 = projectOutDims(std::move(R0), nParams, 1);
            R1 = projectOutDims(std::move(R1), nParams, 1);
            R01 = projectOutDims(std::move(R01), nParams, 1);

            if (!intersect(intersect(R0, R1), complement(R01)).empty())
                break;
        }

        // overlap check 2: There should be no more than 1 dimensions with
        //                  significant non-overlaped interval. Negative
        //                  tolerance is disabling this check.
        if (fusableNonOverlapTolerance >= 0) {
            bool hugeNonOverlap = false;
            auto check = [&](const PBSet &aRange, const PBSet &bRange) {
                if (isl_set_is_bounded(aRange.get())) {
                    auto aMax = PBVal(isl_set_dim_max_val(aRange.copy(), 0));
                    ASSERT(aMax.denSi() == 1);
                    auto offset =
                        fixDim(universeSet(PBSpace(aRange)), 0,
                               aMax.numSi() + fusableNonOverlapTolerance);
                    // $b \cap a + (max(a) + tol) != \emptySet$
                    // indicates b ends long after a
                    if (!intersect(bRange, sum(aRange, offset)).empty())
                        hugeNonOverlap = true;
                    // $b \cap a - (max(a) + tol) != \emptySet$
                    // indicates b begins long before a
                    if (!intersect(bRange, sum(aRange, neg(offset))).empty())
                        hugeNonOverlap = true;
                }
            };
            check(loop0Range, loop1Range);
            check(loop1Range, loop0Range);
            if (hugeNonOverlap) {
                hugeNonOverlapCount++;
                // two dimensions violates, the loop fusion is bloating up,
                // rollback to before first violation and quit
                if (hugeNonOverlapCount == 2) {
                    fusedLevel = hugeNonOverlapFusedLevel;
                    c0ParamValue.resize(fusedLevel);
                    c0IterValue.resize(fusedLevel);
                    c1ParamValue.resize(fusedLevel);
                    c1IterValue.resize(fusedLevel);
                    break;
                }
                hugeNonOverlapFusedLevel = fusedLevel;
            }
        }

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
            optimized.begin() + (nParams + 1) * 3 + nestLevel0 + nestLevel1,
        });
    }

    // if nothing fusable, fail fast
    if (fusedLevel == 0)
        throw InvalidSchedule("No fusable dimension found by Pluto+.");

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

    std::vector<std::string> fusedLoopsVar, remainLoop0Var, remainLoop1Var;
    for (int i = 0; i < fusedLevel; ++i)
        fusedLoopsVar.push_back("fuse_i" + toString(i));
    for (int i = fusedLevel; i < nestLevel0; ++i)
        remainLoop0Var.push_back("rem0_i" + toString(i));
    for (int i = fusedLevel; i < nestLevel1; ++i)
        remainLoop1Var.push_back("rem1_i" + toString(i));

    PermuteInfo loop0Permute(c0ParamValue, c0IterValue, paramExprs, loop0Axes,
                             views::concat(fusedLoopsVar, remainLoop0Var) |
                                 ranges::to<std::vector>(),
                             ctx, loop0Set);
    PermuteInfo loop1Permute(c1ParamValue, c1IterValue, paramExprs, loop1Axes,
                             views::concat(fusedLoopsVar, remainLoop1Var) |
                                 ranges::to<std::vector>(),
                             ctx, loop1Set);

    PlutoFuse fuser(loop0Id, loop1Id, fusedLoopsVar, remainLoop0Var,
                    remainLoop1Var, loop0Permute, loop1Permute);
    ast = fuser(ast);
    ast = shrinkFor(ast, findStmt(ast, fuser.fusedId_), false);
    if (doSimplify)
        ast = pbSimplify(ast);
    if (hoisted)
        ast = sinkVar(ast);

    return {ast, {fuser.fusedId_, parallelCount}};
}

class InjectEmptyLoop : public Mutator {
    ID target_;
    bool clearingBody_;
    ID emptyLoopId_;

  public:
    InjectEmptyLoop(const ID &target) : target_(target), clearingBody_(false) {}

    const ID &emptyLoopId() const { return emptyLoopId_; }

  protected:
    Stmt visit(const For &op) override {
        if (clearingBody_) {
            Stmt body;
            // recurse into next level when it's still a loop
            if (op->body_->nodeType() == ASTNodeType::For)
                body = (*this)(op->body_);
            // nest loops end, clear the body
            else
                body = makeStmtSeq({});
            // reset the id to avoid conflict
            return makeFor(op->iter_, op->begin_, op->end_, op->step_, op->len_,
                           op->property_, body);
        } else if (op->id() == target_) {
            clearingBody_ = true;
            auto emptyLoop = (*this)(op);
            emptyLoopId_ = emptyLoop->id();
            clearingBody_ = false;
            return makeStmtSeq({emptyLoop, op});
        }
        return Mutator::visit(op);
    }
};

bool isAffectedLoop(const For &loop, const ID &baseLoopId, int nestLevel) {
    for (Stmt s = loop; s.isValid(); s = s->parentStmt()) {
        if (s->nodeType() == ASTNodeType::For) {
            if (s->id() == baseLoopId) {
                return true;
            }
            if (nestLevel > 0 /* not limited */ || --nestLevel == 0) {
                return false;
            }
        }
    }
    return false;
}

} // namespace

std::pair<Stmt, std::pair<ID, int>>
plutoFuse(const Stmt &_ast, const ID &loop0Id, const ID &loop1Id,
          int nestLevel0, int nestLevel1, int fusableOverlapThreshold,
          int fusableNonOverlapTolerance, bool doSimplify) {
    auto ast = normalizeLoops(_ast, [&](auto &&l) {
        return isAffectedLoop(l, loop0Id, nestLevel0) ||
               isAffectedLoop(l, loop1Id, nestLevel1);
    });
    ast = flattenStmtSeq(ast);
    return plutoFuseImpl(ast, loop0Id, loop1Id, nestLevel0, nestLevel1,
                         fusableOverlapThreshold, fusableNonOverlapTolerance,
                         doSimplify);
}

std::pair<Stmt, std::pair<ID, int>>
plutoPermute(const Stmt &_ast, const ID &loop, int nestLevel, bool doSimplify) {
    auto ast = normalizeLoops(
        _ast, [&](auto &&l) { return isAffectedLoop(l, loop, nestLevel); });
    InjectEmptyLoop injecter(loop);
    ast = injecter(_ast);
    return plutoFuseImpl(ast, injecter.emptyLoopId(), loop, nestLevel,
                         nestLevel, 1, -1, doSimplify);
}

std::pair<ID, int> Schedule::plutoFuse(const ID &loop0, const ID &loop1,
                                       int nestLevel0, int nestLevel1,
                                       int fusableOverlapThreshold,
                                       int fusableNonOverlapTolerance,
                                       bool doSimplify) {
    beginTransaction();
    auto log = appendLog(MAKE_SCHEDULE_LOG(
        PlutoFuse, freetensor::plutoFuse, loop0, loop1, nestLevel0, nestLevel1,
        fusableOverlapThreshold, fusableNonOverlapTolerance, doSimplify));
    try {
        auto ret = applyLog(log);
        commitTransaction();
        return ret;
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

std::pair<ID, int> Schedule::plutoPermute(const ID &loop, int nestLevel,
                                          bool doSimplify) {
    beginTransaction();
    auto log = appendLog(MAKE_SCHEDULE_LOG(
        PlutoPermute, freetensor::plutoPermute, loop, nestLevel, doSimplify));
    try {
        auto ret = applyLog(log);
        commitTransaction();
        return ret;
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

} // namespace freetensor
