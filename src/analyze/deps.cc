#include <algorithm>
#include <sstream>

#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <container_utils.h>
#include <disjoint_set.h>
#include <except.h>
#include <mutator.h>
#include <omp_utils.h>
#include <pass/const_fold.h>
#include <serialize/mangle.h>
#include <serialize/print_ast.h>

namespace freetensor {

void FindAllNoDeps::visit(const For &op) {
    Visitor::visit(op);
    for (auto &&var : op->property_->noDeps_) {
        results_[var].emplace_back(op->id());
    }
}

FindAccessPoint::FindAccessPoint(const ID &vardef, DepType depType,
                                 const FindDepsAccFilter &accFilter)
    : vardef_(vardef), depType_(depType), accFilter_(accFilter) {}

void FindAccessPoint::doFind(const Stmt &root) {
    // Push potential StmtSeq scope
    cur_.emplace_back(makeIntConst(-1));

    (*this)(root);

    // Pop potential StmtSeq scope
    if (checkTrivialScope(reads_.begin(), reads_.end()) &&
        checkTrivialScope(writes_.begin(), writes_.end())) {
        removeTrivialScopeFromAccesses(reads_.begin(), reads_.end());
        removeTrivialScopeFromAccesses(writes_.begin(), writes_.end());
        removeTrivialScopeFromScopes(allScopes_.begin(), allScopes_.end());
    }
    cur_.pop_back();
}

bool FindAccessPoint::checkTrivialScope(
    std::vector<Ref<AccessPoint>>::iterator begin,
    std::vector<Ref<AccessPoint>>::iterator end) {
    int dim = (int)cur_.size() - 1;
    for (auto it = begin; it != end; it++) {
        auto &&coord = (*it)->iter_.at(dim).iter_;
        ASSERT(coord->nodeType() == ASTNodeType::IntConst);
        if (coord.as<IntConstNode>()->val_ > 0) {
            return false;
        }
    }
    return true;
}

void FindAccessPoint::removeTrivialScopeFromAccesses(
    std::vector<Ref<AccessPoint>>::iterator begin,
    std::vector<Ref<AccessPoint>>::iterator end) {
    int dim = (int)cur_.size() - 1;
    for (auto it = begin; it != end; it++) {
        (*it)->iter_.erase((*it)->iter_.begin() + dim);
        if ((*it)->defAxis_ > dim) {
            (*it)->defAxis_--;
        }
    }
}

void FindAccessPoint::removeTrivialScopeFromScopes(
    std::vector<ID>::iterator begin, std::vector<ID>::iterator end) {
    int dim = (int)cur_.size() - 1;
    for (auto it = begin; it != end; it++) {
        if (auto jt = scope2coord_.find(*it); jt != scope2coord_.end()) {
            auto &coord = jt->second;
            ASSERT(dim < (int)coord.size());
            if (dim + 1 < (int)coord.size()) {
                // The position of this scope is changed
                coord.erase(coord.begin() + dim);
            } else {
                // We no longer have this scope
                scope2coord_.erase(jt);
            }
        }
    }
}

void FindAccessPoint::visit(const VarDef &op) {
    if (op->id() == vardef_) {
        defAxis_ = !cur_.empty() && cur_.back().iter_->nodeType() ==
                                        ASTNodeType::IntConst
                       ? cur_.size() - 1
                       : cur_.size();
        BaseClass::visit(op);
        defAxis_ = -1;
    } else {
        BaseClass::visit(op);
    }
}

void FindAccessPoint::visit(const StmtSeq &op) {
    ASSERT(!cur_.empty());
    scope2coord_[op->id()] = cur_;
    BaseClass::visit(op);
    allScopes_.emplace_back(op->id());
}

void FindAccessPoint::visit(const For &op) {
    (*this)(op->begin_);
    (*this)(op->end_);
    (*this)(op->len_);

    auto old = cur_;
    auto oldLastIsLoad = lastIsLoad_;
    if (!cur_.empty() &&
        cur_.back().iter_->nodeType() == ASTNodeType::IntConst) {
        // top is band node
        cur_.back().iter_ =
            makeIntConst(cur_.back().iter_.as<IntConstNode>()->val_ + 1);
    }
    lastIsLoad_ = false;

    auto iter = makeVar(op->iter_);
    auto oldCondsSize = conds_.size();
    if (auto &&step = constFold(op->step_);
        step->nodeType() == ASTNodeType::IntConst) {
        auto stepVal = step.as<IntConstNode>()->val_;
        if (stepVal > 0) {
            pushCond(
                makeLAnd(
                    makeLAnd(makeGE(iter, op->begin_), makeLT(iter, op->end_)),
                    makeEQ(makeMod(makeSub(iter, op->begin_), op->step_),
                           makeIntConst(0))),
                op->id());
            cur_.emplace_back(iter, op->property_->parallel_);
        } else if (stepVal == 0) {
            pushCond(makeEQ(iter, op->begin_), op->id());
            cur_.emplace_back(iter, op->property_->parallel_);
        } else {
            pushCond(
                makeLAnd(
                    makeLAnd(makeLE(iter, op->begin_), makeGT(iter, op->end_)),
                    makeEQ(makeMod(makeSub(iter, op->begin_), op->step_),
                           makeIntConst(0))),
                op->id());
            cur_.emplace_back(iter, op->property_->parallel_, true);
        }
    } else {
        ERROR("Currently loops with an unknown sign of step is not supported "
              "in analyze/deps");
    }

    // Push For scope
    ASSERT(!cur_.empty());
    scope2coord_[op->id()] = cur_;

    // Push potential StmtSeq scope
    cur_.emplace_back(makeIntConst(-1));
    auto oldReadsSize = reads_.size();
    auto oldWritesSize = writes_.size();
    auto oldAllScopesSize = allScopes_.size();

    pushFor(op);
    (*this)(op->body_);
    popFor(op);

    // Pop potential StmtSeq scope
    auto oldReadsEnd = reads_.begin() + oldReadsSize;
    auto oldWritesEnd = writes_.begin() + oldWritesSize;
    auto oldAllScopesEnd = allScopes_.begin() + oldAllScopesSize;
    if (checkTrivialScope(oldReadsEnd, reads_.end()) &&
        checkTrivialScope(oldWritesEnd, writes_.end())) {
        removeTrivialScopeFromAccesses(oldReadsEnd, reads_.end());
        removeTrivialScopeFromAccesses(oldWritesEnd, writes_.end());
        removeTrivialScopeFromScopes(oldAllScopesEnd, allScopes_.end());
    }
    cur_.pop_back();

    // Pop For scope
    cur_.pop_back();
    conds_.resize(oldCondsSize);
    allScopes_.emplace_back(op->id());

    lastIsLoad_ = false; // The last Load in the loop and the first Load out of
                         // the loop shall have different coordinates

    if (oldReadsEnd == reads_.end() && oldWritesEnd == writes_.end()) {
        // No stepping to make iteration space more compact
        cur_ = std::move(old);
        lastIsLoad_ = oldLastIsLoad;
        scope2coord_.erase(op->id());
    }
}

void FindAccessPoint::visit(const If &op) {
    (*this)(op->cond_);

    if (!op->elseCase_.isValid()) {
        pushCond(op->cond_, op->id());
        (*this)(op->thenCase_);
        popCond();
    } else {
        pushCond(op->cond_, op->id());
        (*this)(op->thenCase_);
        popCond();
        pushCond(makeLNot(op->cond_), op->id());
        (*this)(op->elseCase_);
        popCond();
    }
}

void FindAccessPoint::visit(const Assert &op) {
    (*this)(op->cond_);

    pushCond(op->cond_, op->id());
    (*this)(op->body_);
    popCond();
}

void FindAccessPoint::visit(const Load &op) {
    BaseClass::visit(op);

    if (!(depType_ & DEP_RAW) && !(depType_ & DEP_WAR)) {
        return;
    }

    bool isThisVarDef = false;
    VarDef viewOf;
    if (def(op->var_)->id() == vardef_) {
        isThisVarDef = true;
    } else {
        for (auto source = def(op->var_); source->viewOf_.has_value();) {
            source = def(*source->viewOf_);
            if (source->id() == vardef_) {
                isThisVarDef = true;
                viewOf = source;
                break;
            }
        }
    }
    if (!isThisVarDef) {
        return;
    }

    std::vector<Expr> exprs;
    VarDef d;
    if (viewOf.isValid()) {
        // Simultaneously access of a `VarDef` and the `VarDef` it views is
        // ALWAYS treated as dependences. Use Intrinsic as "any expression"
        exprs =
            std::vector<Expr>(viewOf->buffer_->tensor()->shape().size(),
                              makeIntrinsic("", {}, DataType::Int32, false));
        d = viewOf;
    } else {
        exprs = op->indices_;
        d = def(op->var_);
    }

    if (accFilter_ == nullptr ||
        accFilter_(Access{op, curStmt(), d, d->buffer_})) {
        if (!cur_.empty() &&
            cur_.back().iter_->nodeType() == ASTNodeType::IntConst) {
            // top is band node
            if (!lastIsLoad_) {
                cur_.back().iter_ = makeIntConst(
                    cur_.back().iter_.as<IntConstNode>()->val_ + 1);
            }
        }
        lastIsLoad_ = true;

        auto ap = Ref<AccessPoint>::make();
        *ap = {op,   curStmt(),        d,     d->buffer_, defAxis_,
               cur_, std::move(exprs), conds_};
        reads_.emplace_back(ap);
    }
}

std::string AnalyzeDeps::makeIterList(const std::vector<IterAxis> &list,
                                      int n) {
    std::string ret;
    for (int i = 0; i < n; i++) {
        if (i < (int)list.size()) {
            if (list[i].iter_->nodeType() == ASTNodeType::Var) {
                ret += mangle(list[i].iter_.as<VarNode>()->name_);
            } else if (list[i].iter_->nodeType() == ASTNodeType::IntConst) {
                ret += std::to_string(list[i].iter_.as<IntConstNode>()->val_);
            } else {
                ASSERT(false);
            }
        } else {
            ret += "0";
        }
        if (i < n - 1) {
            ret += ", ";
        }
    }
    return "[" + ret + "]";
}

std::string AnalyzeDeps::makeNegIterMap(const std::vector<IterAxis> &list,
                                        int n) {
    std::string lhs, rhs;
    for (int i = 0; i < n; i++) {
        if (i < (int)list.size() && list[i].negStep_) {
            rhs += "-";
        }
        auto name = "i" + std::to_string(i);
        lhs += name;
        rhs += name;
        if (i < n - 1) {
            lhs += ", ";
            rhs += ", ";
        }
    }
    return "{[" + lhs + "] -> [" + rhs + "]}";
}

std::string AnalyzeDeps::makeAccList(GenPBExpr &genPBExpr,
                                     const std::vector<Expr> &list,
                                     RelaxMode relax,
                                     GenPBExpr::VarMap &externals) {
    std::string ret;
    for (int i = 0, iEnd = list.size(); i < iEnd; i++) {
        auto &&[linstr, vars] = genPBExpr.gen(list[i]);
        ret += linstr;
        for (auto &&[expr, str] : vars) {
            if (expr->nodeType() != ASTNodeType::Var) {
                externals[expr] = str;
            }
        }
        if (i < iEnd - 1) {
            ret += ", ";
        }
    }
    return "[" + ret + "]";
}

std::string AnalyzeDeps::makeCond(GenPBExpr &genPBExpr,
                                  const std::vector<std::pair<Expr, ID>> &conds,
                                  RelaxMode relax, GenPBExpr::VarMap &externals,
                                  bool eraseOutsideVarDef,
                                  const AccessPoint &ap) {
    // If the condition is defined outside of the variable we are analyzing, and
    // external variables in this condition is not used in any accessing or
    // iterating coordinates, and if `eraseOutsideVarDef_` is enabled, we can
    // safely ignore this condition, because all access of this variable will
    // hold the same condition value
    std::vector<bool> isRedundants(conds.size(), false);
    if (eraseOutsideVarDef) {
        auto namesInConds =
            conds |
            views::transform([](auto &&x) { return allNames(x.first, true); }) |
            ranges::to_vector;

        DisjointSet<std::string> namesConnectivity;
        // add a root for actually used names
        namesConnectivity.find("");
        // mark names from both access and loop indices as used
        for (auto &&idx : ap.access_)
            for (auto &&name : allNames(idx, true))
                namesConnectivity.uni("", name);
        for (auto &&iter : ap.iter_)
            for (auto &&name : allNames(iter.iter_, true))
                namesConnectivity.uni("", name);

        // connect by conditions
        for (auto &&[condItem, names] : views::zip(conds, namesInConds)) {
            auto &&[_, condId] = condItem;

            std::optional<std::string> first;

            // if this condition is not defined outside, treat all its
            // corresponded names as used
            if (!ap.def_->ancestorById(condId).isValid())
                first = "";

            // connect the names occurred
            for (auto &&name : names)
                if (first.has_value())
                    namesConnectivity.uni(*first, name);
                else
                    first = name;
        }

        auto root = namesConnectivity.find("");
        for (size_t i = 0; i < conds.size(); ++i) {
            isRedundants[i] = true;
            for (auto &&name : namesInConds[i])
                if (namesConnectivity.find(name) == root) {
                    isRedundants[i] = false;
                    break;
                }
        }
    }

    std::string ret;
    for (auto &&[condItem, isRedundant] : views::zip(conds, isRedundants)) {
        if (isRedundant) {
            continue;
        }

        auto &&[cond, baseStmtId] = condItem;
        auto &&[str, vars] = genPBExpr.gen(cond);
        for (auto &&[expr, str] : vars) {
            if (expr->nodeType() != ASTNodeType::Var) {
                externals[expr] = str;
            }
        }
        if (!ret.empty()) {
            ret += " and ";
        }
        ret += str;
    }

    return ret;
}

PBMap AnalyzeDeps::makeAccMap(PBCtx &presburger, const AccessPoint &p,
                              int iterDim, int accDim, RelaxMode relax,
                              const std::string &extSuffix,
                              GenPBExpr::VarMap &externals,
                              const ASTHashSet<Expr> &noNeedToBeVars) {
    GenPBExpr genPBExpr(extSuffix, noNeedToBeVars);
    auto ret = makeIterList(p.iter_, iterDim) + " -> " +
               makeAccList(genPBExpr, p.access_, relax, externals);
    if (auto str = makeCond(genPBExpr, p.conds_, relax, externals,
                            eraseOutsideVarDef_, p);
        !str.empty()) {
        ret += ": " + str;
    }
    std::string ext;
    if (!externals.empty()) {
        bool first = true;
        for (auto &&[expr, str] : externals) {
            ext += (first ? "" : ", ") + str;
            first = false;
        }
        ext = "[" + ext + "] -> ";
    }
    ret = ext + "{" + ret + "}";
    auto unordered = PBMap(presburger, ret);
    auto negIterMap = PBMap(presburger, makeNegIterMap(p.iter_, iterDim));
    auto ordered = applyDomain(std::move(unordered), std::move(negIterMap));
    return ordered;
}

std::string AnalyzeDeps::makeNdList(const std::string &name, int n) {
    std::string ret;
    for (int i = 0; i < n; i++) {
        ret += name + std::to_string(i);
        if (i < n - 1) {
            ret += ",";
        }
    }
    return "[" + ret + "]";
}

PBMap AnalyzeDeps::makeEqForBothOps(
    PBCtx &presburger, const std::vector<std::pair<int, int>> &coord,
    int iterDim) const {
    auto map = universeMap(spaceAlloc(presburger, 0, iterDim, iterDim)).move();
    for (auto &&[dim, val] : coord)
        map = isl_map_fix_si(isl_map_fix_si(map, isl_dim_out, dim, val),
                             isl_dim_in, dim, val);
    return PBMap(map);
}

PBMap AnalyzeDeps::makeIneqBetweenOps(PBCtx &presburger, DepDirection mode,
                                      int iterId, int iterDim) const {
    switch (mode) {
    case DepDirection::Inv:
        return PBMap(isl_map_order_gt(
            universeMap(spaceAlloc(presburger, 0, iterDim, iterDim)).move(),
            isl_dim_out, iterId, isl_dim_in, iterId));
    case DepDirection::Normal:
        return PBMap(isl_map_order_lt(
            universeMap(spaceAlloc(presburger, 0, iterDim, iterDim)).move(),
            isl_dim_out, iterId, isl_dim_in, iterId));
    case DepDirection::Same:
        return PBMap(isl_map_equate(
            universeMap(spaceAlloc(presburger, 0, iterDim, iterDim)).move(),
            isl_dim_out, iterId, isl_dim_in, iterId));
    case DepDirection::Different:
        return uni(
            PBMap(isl_map_order_lt(
                universeMap(spaceAlloc(presburger, 0, iterDim, iterDim)).move(),
                isl_dim_out, iterId, isl_dim_in, iterId)),
            PBMap(isl_map_order_gt(
                universeMap(spaceAlloc(presburger, 0, iterDim, iterDim)).move(),
                isl_dim_out, iterId, isl_dim_in, iterId)));
    default:
        ASSERT(false);
    }
}

PBMap AnalyzeDeps::makeConstraintOfSingleLoop(PBCtx &presburger, const ID &loop,
                                              DepDirection mode, int iterDim) {
    if (!scope2coord_.count(loop)) {
        // If we don't have the scope in `scope2coord_`, it means the scope is
        // trivial, which has only one instance inside. So it must be `Same`
        if (mode == DepDirection::Same) {
            return universeMap(spaceAlloc(presburger, 0, iterDim, iterDim));
        } else {
            return emptyMap(spaceAlloc(presburger, 0, iterDim, iterDim));
        }
    }

    auto &&coord = scope2coord_.at(loop);
    ASSERT(coord.size() > 0); // Or we would have removed it from scope2coord_
    int iterId = coord.size() - 1;
    if (iterId >= iterDim) {
        return emptyMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    }

    auto ret = universeMap(spaceAlloc(presburger, 0, iterDim, iterDim));

    // Position in the outer StmtSeq nodes
    std::vector<std::pair<int, int>> pos;
    for (int i = 0; i < iterId; i++) {
        if (coord[i].iter_->nodeType() == ASTNodeType::IntConst) {
            pos.emplace_back(i, coord[i].iter_.as<IntConstNode>()->val_);
        }
    }
    if (!pos.empty()) {
        ret = intersect(std::move(ret),
                        makeEqForBothOps(presburger, pos, iterDim));
    }

    return intersect(std::move(ret),
                     makeIneqBetweenOps(presburger, mode, iterId, iterDim));
}

PBMap AnalyzeDeps::makeConstraintOfParallelScope(PBCtx &presburger,
                                                 const ParallelScope &parallel,
                                                 DepDirection mode, int iterDim,
                                                 const AccessPoint &later,
                                                 const AccessPoint &earlier) {
    int laterDim = -1, earlierDim = -1;
    for (int i = (int)later.iter_.size() - 1; i >= 0; i--) {
        if (later.iter_[i].parallel_ == parallel) {
            laterDim = i;
            break;
        }
    }
    for (int i = (int)earlier.iter_.size() - 1; i >= 0; i--) {
        if (earlier.iter_[i].parallel_ == parallel) {
            earlierDim = i;
            break;
        }
    }
    if (earlierDim == -1 && laterDim == -1) {
        return emptyMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    }
    if (earlierDim == -1 || laterDim == -1) {
        return universeMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    }

    std::string ineq;
    switch (mode) {
    case DepDirection::Inv:
        ineq = ">";
        break;
    case DepDirection::Normal:
        ineq = "<";
        break;
    case DepDirection::Same:
        ineq = "=";
        break;
    case DepDirection::Different:
        ineq = "!=";
        break;
    default:
        ASSERT(false);
    }
    // FIXME: parallel loop of the same parallel scope of later and earlier may
    // have different `begin`, we must substract `begin` before compareing
    return PBMap(presburger, "{" + makeNdList("d", iterDim) + " -> " +
                                 makeNdList("d_", iterDim) + ": d_" +
                                 std::to_string(earlierDim) + " " + ineq +
                                 " d" + std::to_string(laterDim) + "}");
}

PBMap AnalyzeDeps::makeExternalEq(PBCtx &presburger, int iterDim,
                                  const std::string &ext1,
                                  const std::string &ext2) {
    PBMap universe = universeMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    PBSet constraint =
        PBSet(presburger, "[" + ext1 + ", " + ext2 + "] -> {[] : " + ext1 +
                              " = " + ext2 + "}");
    return intersectParams(universe, constraint);
}

const std::string &AnalyzeDeps::getVar(const AST &op) {
    switch (op->nodeType()) {
    case ASTNodeType::Load:
        return op.as<LoadNode>()->var_;
    case ASTNodeType::Store:
        return op.as<StoreNode>()->var_;
    case ASTNodeType::ReduceTo:
        return op.as<ReduceToNode>()->var_;
    default:
        ASSERT(false);
    }
}

PBMap AnalyzeDeps::makeSerialToAll(PBCtx &presburger, int iterDim,
                                   const std::vector<IterAxis> &point) const {
    std::string to = makeNdList("d", iterDim), from;
    for (int i = 0; i < iterDim; i++) {
        if (i < (int)point.size() && point[i].parallel_ != serialScope) {
            from += std::string(i > 0 ? ", " : "") + "0";
        } else {
            from += std::string(i > 0 ? ", " : "") + "d" + std::to_string(i);
        }
    }
    from = "[" + from + "]";
    return PBMap(presburger, "{" + from + " -> " + to + "}");
}

PBMap AnalyzeDeps::makeEraseVarDefConstraint(PBCtx &presburger,
                                             const Ref<AccessPoint> &point,
                                             int iterDim) {
    PBMap ret = universeMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    if (eraseOutsideVarDef_) {
        for (int i = 0; i < point->defAxis_; i++) {
            ret = intersect(
                std::move(ret),
                makeIneqBetweenOps(presburger, DepDirection::Same, i, iterDim));
        }
    }
    return ret;
}

PBMap AnalyzeDeps::makeNoDepsConstraint(PBCtx &presburger,
                                        const std::string &var, int iterDim) {
    PBMap ret = universeMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    if (noDepsLists_.count(var)) {
        for (auto &&noDepsLoop : noDepsLists_.at(var)) {
            auto noDep = makeConstraintOfSingleLoop(
                presburger, noDepsLoop, DepDirection::Different, iterDim);
            ret = subtract(std::move(ret), std::move(noDep));
        }
    }
    return ret;
}

PBMap AnalyzeDeps::makeExternalVarConstraint(
    PBCtx &presburger, const Ref<AccessPoint> &later,
    const Ref<AccessPoint> &earlier, const GenPBExpr::VarMap &laterExternals,
    const GenPBExpr::VarMap &earlierExternals, int iterDim) {
    PBMap ret = universeMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    // We only have to add constraint for common loops of both accesses
    auto common = lcaStmt(later->stmt_, earlier->stmt_);

    for (auto &&expr : ranges::to<ASTHashSet<Expr>>(views::concat(
             views::keys(laterExternals), views::keys(earlierExternals)))) {
        auto &&pStr = mangle(dumpAST(expr, true)) + "__ext__later" +
                      std::to_string((uint64_t)later->stmt_->id());
        auto &&oStr = mangle(dumpAST(expr, true)) + "__ext__earlier" +
                      std::to_string((uint64_t)earlier->stmt_->id());
        auto require = makeExternalEq(presburger, iterDim, pStr, oStr);
        for (auto c = common; c.isValid(); c = c->parentStmt()) {
            if (eraseOutsideVarDef_ && c->id() == later->def_->id()) {
                // Quick path: If eraseOutsideVarDef_ enabled, we will enforce
                // outer loops to be in the same iteration, so no need to do
                // `require = require || (in different iterations)` below
                break;
            }
            if (c->nodeType() == ASTNodeType::For) {
                // An external expresson is invariant to a loop if:
                // 1. The expression in earlier is invariant to the loop, and
                // 2. The expression is not modified in the loop, and
                // 3. (1 and 2 imply) the expression in later is also invariant
                bool invariant = !isVariant(*variantExpr_,
                                            {expr, earlier->stmt_}, c->id()) &&
                                 !hasIntersect(allReads(expr), allWrites(c));
                if (!invariant) {
                    // Since idx[i] must be inside loop i, we only have
                    // to call makeIneqBetweenOps, but no need to call
                    // makeConstraintOfSingleLoop
                    auto diffIter = makeIneqBetweenOps(
                        presburger, DepDirection::Different,
                        scope2coord_.at(c->id()).size() - 1, iterDim);
                    require = uni(std::move(diffIter), std::move(require));
                }
            }
        }
        ret = intersect(std::move(ret), std::move(require));
    }
    return ret;
}

PBMap AnalyzeDeps::projectOutPrivateAxis(PBCtx &presburger, int iterDim,
                                         int since) {
    std::string from = makeNdList("d", iterDim);
    std::string to;
    for (int i = 0; i < iterDim; i++) {
        to += (i > 0 ? ", " : "") + (i < since ? "d" + std::to_string(i) : "0");
    }
    to = "[" + to + "]";
    return PBMap(presburger, "{" + from + " -> " + to + "}");
}

void AnalyzeDeps::projectOutPrivateAxis(
    PBCtx &presburger, const Ref<AccessPoint> &point,
    const std::vector<Ref<AccessPoint>> &otherList,
    std::vector<PBMap> &otherMapList, int iterDim) {
    if (!noProjectOutPrivateAxis_) {
        std::vector<int> oCommonDims(otherList.size(), 0);
        for (size_t i = 0, n = otherList.size(); i < n; i++) {
            auto &&other = otherList[i];
            int cpo = numCommonDims(point, other);
            oCommonDims[i] = std::max(oCommonDims[i], cpo);
            if (i + 1 < n) {
                int co1o2 = numCommonDims(other, otherList[i + 1]);
                oCommonDims[i] = std::max(oCommonDims[i], co1o2);
                oCommonDims[i + 1] = std::max(oCommonDims[i + 1], co1o2);
            }
        }

        for (auto &&[common, other, otherMap] :
             views::zip(oCommonDims, otherList, otherMapList)) {
            if (common + 1 < (int)other->iter_.size()) {
                otherMap = applyDomain(
                    std::move(otherMap),
                    projectOutPrivateAxis(presburger, iterDim, common + 1));
                otherMap = coalesce(std::move(otherMap));
            }
        }
    }
}

int AnalyzeDeps::numCommonDims(const Ref<AccessPoint> &p1,
                               const Ref<AccessPoint> &p2) {
    int n = std::min(p1->iter_.size(), p2->iter_.size());
    for (int i = 0; i < n; i++) {
        auto &&iter1 = p1->iter_[i].iter_;
        auto &&iter2 = p2->iter_[i].iter_;
        if (iter1->nodeType() == ASTNodeType::IntConst &&
            iter2->nodeType() == ASTNodeType::IntConst &&
            iter1.as<IntConstNode>()->val_ != iter2.as<IntConstNode>()->val_) {
            for (int j = n - 1; j >= i; j--) {
                if ((j < (int)p1->iter_.size() &&
                     p1->iter_[j].parallel_ != serialScope) ||
                    (j < (int)p2->iter_.size() &&
                     p2->iter_[j].parallel_ != serialScope)) {
                    return j + 1;
                }
            }
            return i;
        }
    }
    return n;
}

void AnalyzeDeps::checkAgainstCond(PBCtx &presburger,
                                   const Ref<AccessPoint> &later,
                                   const Ref<AccessPoint> &earlier,
                                   const PBMap &depAll, const PBMap &nearest,
                                   const PBMap &laterMap,
                                   const PBMap &earlierMap,
                                   const PBMap &extConstraint, int iterDim) {
    if (nearest.empty()) {
        return;
    }

    if (mode_ != FindDepsMode::Dep) {
        // For dependence X->Y or Y->X, we can check for kill-X, which means any
        // time (any coordinate in iteration space and any external variable of
        // X) there is X, there is the dependence.
        //
        // Here "any time" does not include impossible external varaible
        // combinations ruled out by `extConstraint`, so we need to intersect
        // with it before checking.
        //
        // Besides, (TODO: prove it) `extConstraint` includes multiple cases,
        // which can be written as `{[...] -> [...] : coordinates 1 -> params
        // constraints 1, or coordinates 2 -> params constraints 2, or ...}`. We
        // intersect with `nearest`'s coordinates to get the effective parameter
        // constraints.
        auto effectiveExtConstraint =
            params(intersect(extConstraint, projectOutAllParams(nearest)));
        PBSet realEarlierIter, realLaterIter;
        if (mode_ == FindDepsMode::KillEarlier ||
            mode_ == FindDepsMode::KillBoth) {
            realEarlierIter =
                intersectParams(domain(earlierMap), effectiveExtConstraint);
        }
        if (mode_ == FindDepsMode::KillLater ||
            mode_ == FindDepsMode::KillBoth) {
            realLaterIter =
                intersectParams(domain(laterMap), effectiveExtConstraint);
        }

        // Range/domain of `nearest` is always a subset of the iterating space
        // of `earlierMap`/`laterMap`, and we want to check whether it is a
        // strict subset. Since internally eiter `isl_set_is_strict_subset` or
        // `isl_set_is_equal` is implemented by two `isl_set_is_subset`s (in
        // both direction), we only need to check one of them:
        // `isStrictSubset(...nearest, ...earlierMap) =>
        // !isSubset(...earlierMap, ...nearest)`.

        // Coarse check (depAll is a superset of nearest)
        if ((mode_ == FindDepsMode::KillEarlier ||
             mode_ == FindDepsMode::KillBoth) &&
            !isSubset(realEarlierIter, range(depAll))) {
            return;
        }
        if ((mode_ == FindDepsMode::KillLater ||
             mode_ == FindDepsMode::KillBoth) &&
            !isSubset(realLaterIter, domain(depAll))) {
            return;
        }

        // Fine check
        if ((mode_ == FindDepsMode::KillEarlier ||
             mode_ == FindDepsMode::KillBoth) &&
            !isSubset(realEarlierIter, range(nearest))) {
            return;
        }
        if ((mode_ == FindDepsMode::KillLater ||
             mode_ == FindDepsMode::KillBoth) &&
            !isSubset(realLaterIter, domain(nearest))) {
            return;
        }
    }

    for (auto &&item : direction_) {
        std::vector<PBMap> _requires;
        for (auto &&[nodeOrParallel, dir] : item) {
            if (nodeOrParallel.isNode_) {
                _requires.emplace_back(makeConstraintOfSingleLoop(
                    presburger, nodeOrParallel.id_, dir, iterDim));
            } else {
                _requires.emplace_back(makeConstraintOfParallelScope(
                    presburger, nodeOrParallel.parallel_, dir, iterDim, *later,
                    *earlier));
            }
        }

        // Early exit: if there is no intersection on `depAll`, there must
        // be no intersection on `nearest`. Computing on `nearest` is much
        // heavier because it contains more basic maps
        PBMap res = nearest, possible = depAll;
        for (auto &&require : _requires) {
            possible = intersect(std::move(possible), require);
            if (possible.empty()) {
                goto fail;
            }
        }

        for (auto &&require : _requires) {
            res = intersect(std::move(res), std::move(require));
            if (res.empty()) {
                goto fail;
            }
        }
        if (noProjectOutPrivateAxis_) {
            found_(Dependence{item, getVar(later->op_), *later, *earlier,
                              iterDim, res, laterMap, earlierMap, possible,
                              extConstraint, presburger, *this});
        } else {
            // It will be misleading if we pass Presburger maps to users in
            // this case
            found_(Dependence{item, getVar(later->op_), *later, *earlier,
                              iterDim, PBMap(), PBMap(), PBMap(), PBMap(),
                              PBMap(), presburger, *this});
        }
    fail:;
    }
}

static ASTHashSet<Expr>
getNoNeedToBeVars(const std::vector<Ref<AccessPoint>> accesses) {
    // If an external variables is always used inside a fixed expression, we can
    // represent the whole expression as an external variable, to reduce the
    // number of external varaibles. E.g., consider a range of a loop variable
    // is `0 <= i < n[] * m[]`, if `n[]` and `m[]` are always used as `n[] *
    // m[]`, we can simply represent the range as `0 <= i < x`, where `x` equals
    // to `n[] * m[]`. We sum the occurence of each (sub-)expression, to check
    // for this case

    auto checkAllExprs = [&](auto &&callback) {
        for (auto &&acc : accesses) {
            for (auto &&axis : acc->iter_) {
                callback(axis.iter_);
            }
            for (auto &&idx : acc->access_) {
                callback(idx);
            }
            for (auto &&[cond, _] : acc->conds_) {
                callback(cond);
            }
        }
    };

    ASTHashMap<Expr, int> useCnt;
    ASTHashSet<Expr> noNeedToBeVars;

    class SumUseCnt : public Visitor {
        ASTHashMap<Expr, int> &useCnt_;

      public:
        SumUseCnt(ASTHashMap<Expr, int> &useCnt) : useCnt_(useCnt) {}

      protected:
        void visitExpr(const Expr &expr) {
            Visitor::visitExpr(expr);
            useCnt_[expr]++;
        }
    };
    checkAllExprs(SumUseCnt(useCnt));

    class CheckNoNeedToBeVars : public Visitor {
        const ASTHashMap<Expr, int> &useCnt_;
        ASTHashSet<Expr> &noNeedToBeVars_;

      public:
        CheckNoNeedToBeVars(const ASTHashMap<Expr, int> &useCnt,
                            ASTHashSet<Expr> &noNeedToBeVars)
            : useCnt_(useCnt), noNeedToBeVars_(noNeedToBeVars) {}

      protected:
        void visitExpr(const Expr &expr) {
            Visitor::visitExpr(expr);
            for (auto &&child : expr->children()) {
                if (child->nodeType() == ASTNodeType::Var ||
                    useCnt_.at(child) > useCnt_.at(expr)) {
                    return;
                }
            }
            for (auto &&child : expr->children()) {
                noNeedToBeVars_.insert(child);
            }
        }
    };
    checkAllExprs(CheckNoNeedToBeVars(useCnt, noNeedToBeVars));

    return noNeedToBeVars;
}

void AnalyzeDeps::checkDepLatestEarlier(
    const Ref<AccessPoint> &later,
    const std::vector<Ref<AccessPoint>> &_earlierList) {
    std::vector<Ref<AccessPoint>> earlierList;
    for (auto &&earlier : _earlierList) {
        if (ignoreReductionWAW_ &&
            later->op_->nodeType() == ASTNodeType::ReduceTo &&
            earlier->op_->nodeType() == ASTNodeType::ReduceTo) {
            continue;
        }
        if (filter_ == nullptr || filter_(*later, *earlier)) {
            earlierList.emplace_back(earlier);
        }
    }
    if (earlierList.empty()) {
        return;
    }
    tasks_.emplace_back([later, earlierList = std::move(earlierList), this]() {
        PBCtx presburger;
        checkDepLatestEarlierImpl(presburger, later, earlierList);
    });
}

void AnalyzeDeps::checkDepEarliestLater(
    const std::vector<Ref<AccessPoint>> &_laterList,
    const Ref<AccessPoint> &earlier) {
    std::vector<Ref<AccessPoint>> laterList;
    for (auto &&later : _laterList) {
        if (ignoreReductionWAW_ &&
            later->op_->nodeType() == ASTNodeType::ReduceTo &&
            earlier->op_->nodeType() == ASTNodeType::ReduceTo) {
            continue;
        }
        if (filter_ == nullptr || filter_(*later, *earlier)) {
            laterList.emplace_back(later);
        }
    }
    if (laterList.empty()) {
        return;
    }
    tasks_.emplace_back([laterList = std::move(laterList), earlier, this]() {
        PBCtx presburger;
        checkDepEarliestLaterImpl(presburger, laterList, earlier);
    });
}

void AnalyzeDeps::checkDepLatestEarlierImpl(
    PBCtx &presburger, const Ref<AccessPoint> &later,
    const std::vector<Ref<AccessPoint>> &earlierList) {
    int accDim = later->access_.size();
    int iterDim = later->iter_.size();
    for (auto &&earlier : earlierList) {
        iterDim = std::max<int>(iterDim, earlier->iter_.size());
        ASSERT((int)earlier->access_.size() == accDim);
    }

    auto noNeedToBeVars = getNoNeedToBeVars(cat({later}, earlierList));

    PBMap allEQ = identity(spaceAlloc(presburger, 0, iterDim, iterDim));
    PBMap eraseVarDefConstraint =
        makeEraseVarDefConstraint(presburger, later, iterDim);
    PBMap noDepsConstraint =
        makeNoDepsConstraint(presburger, later->def_->name_, iterDim);

    GenPBExpr::VarMap laterExternals;
    PBMap laterMap =
        makeAccMap(presburger, *later, iterDim, accDim, laterRelax_,
                   "later" + std::to_string((uint64_t)later->stmt_->id()),
                   laterExternals, noNeedToBeVars);
    if (laterMap.empty()) {
        return;
    }
    PBMap ls2a = makeSerialToAll(presburger, iterDim, later->iter_);
    PBMap la2s = reverse(ls2a);
    std::vector<PBMap> earlierMapList(earlierList.size());
    std::vector<GenPBExpr::VarMap> earlierExternalsList(earlierList.size());
    std::vector<PBMap> es2aList(earlierList.size()),
        depAllList(earlierList.size());
    PBMap extConstraint, psDepAllUnion;
    for (auto &&[i, earlier, earlierMap, earlierExternals] :
         views::zip(views::ints(0, ranges::unreachable), earlierList,
                    earlierMapList, earlierExternalsList)) {
        earlierMap = makeAccMap(
            presburger, *earlier, iterDim, accDim, earlierRelax_,
            "earlier" + std::to_string((uint64_t)earlier->stmt_->id()),
            earlierExternals, noNeedToBeVars);
    }
    projectOutPrivateAxis(presburger, later, earlierList, earlierMapList,
                          iterDim);
    for (auto &&[i, earlier, earlierMap, earlierExternals, es2a, depAll] :
         views::zip(views::ints(0, ranges::unreachable), earlierList,
                    earlierMapList, earlierExternalsList, es2aList,
                    depAllList)) {
        if (earlierMap.empty()) {
            continue;
        }
        es2a = makeSerialToAll(presburger, iterDim, earlier->iter_);
        PBMap ea2s = reverse(es2a);

        depAll = subtract(applyRange(laterMap, reverse(earlierMap)), allEQ);

        // Constraints on dependences. They decide when there are dependence,
        // but do not affect the existence of accesses. They should be applied
        // only to `depAll`.
        depAll = intersect(std::move(depAll), eraseVarDefConstraint);
        depAll = intersect(std::move(depAll), noDepsConstraint);
        depAll = coalesce(std::move(depAll));

        // Constraints on external variables limit the possibility of certain
        // relations on those variables. If these constraints are violated, it
        // not only means that there is no dependence, but also that the
        // accesses are completely impossible. Therefore, they should be
        // considered as constraints on the universal set, applied to not only
        // `depAll`, but also `earlierMap` and `laterMap` when checking killing
        // cases.
        auto extConstraintLocal = makeExternalVarConstraint(
            presburger, later, earlier, laterExternals, earlierExternals,
            iterDim);
        extConstraint = extConstraint.isValid()
                            ? intersect(std::move(extConstraint),
                                        std::move(extConstraintLocal))
                            : std::move(extConstraintLocal);
        extConstraint = coalesce(std::move(extConstraint));

        PBMap psDepAll = applyRange(depAll, std::move(ea2s));
        psDepAllUnion = psDepAllUnion.isValid()
                            ? uni(std::move(psDepAllUnion), std::move(psDepAll))
                            : std::move(psDepAll);
    }
    if (!psDepAllUnion.isValid()) {
        return;
    }
    psDepAllUnion =
        coalesce(intersect(std::move(psDepAllUnion), extConstraint));

    PBMap serialLexGT = lexGT(spaceSetAlloc(presburger, 0, iterDim));
    PBMap serialEQ = identity(spaceAlloc(presburger, 0, iterDim, iterDim));
    PBMap ssDepAll = applyRange(std::move(ls2a), psDepAllUnion);
    PBMap ssDep = intersect(ssDepAll, std::move(serialLexGT));
    PBMap ssSelf = intersect(ssDepAll, std::move(serialEQ));
    PBMap psDep =
        coalesce(intersect(applyRange(la2s, std::move(ssDep)), psDepAllUnion));
    PBMap psSelf = intersect(applyRange(std::move(la2s), std::move(ssSelf)),
                             std::move(psDepAllUnion));
    PBMap psNearest = uni(lexmax(std::move(psDep)), std::move(psSelf));
    psNearest = coalesce(std::move(psNearest));

    for (auto &&[earlier, es2a, earlierMap, depAll] :
         views::zip(earlierList, es2aList, earlierMapList, depAllList)) {
        if (depAll.isValid()) {
            checkAgainstCond(
                presburger, later, earlier, depAll,
                intersect(applyRange(psNearest, std::move(es2a)), depAll),
                laterMap, earlierMap, extConstraint, iterDim);
        }
    }
}

void AnalyzeDeps::checkDepEarliestLaterImpl(
    PBCtx &presburger, const std::vector<Ref<AccessPoint>> &laterList,
    const Ref<AccessPoint> &earlier) {
    int accDim = earlier->access_.size();
    int iterDim = earlier->iter_.size();
    for (auto &&later : laterList) {
        iterDim = std::max<int>(iterDim, later->iter_.size());
        ASSERT((int)later->access_.size() == accDim);
    }

    auto noNeedToBeVars = getNoNeedToBeVars(cat(laterList, {earlier}));

    PBMap allEQ = identity(spaceAlloc(presburger, 0, iterDim, iterDim));
    PBMap eraseVarDefConstraint =
        makeEraseVarDefConstraint(presburger, earlier, iterDim);
    PBMap noDepsConstraint =
        makeNoDepsConstraint(presburger, earlier->def_->name_, iterDim);

    GenPBExpr::VarMap earlierExternals;
    PBMap earlierMap =
        makeAccMap(presburger, *earlier, iterDim, accDim, earlierRelax_,
                   "earlier" + std::to_string((uint64_t)earlier->stmt_->id()),
                   earlierExternals, noNeedToBeVars);
    if (earlierMap.empty()) {
        return;
    }
    PBMap es2a = makeSerialToAll(presburger, iterDim, earlier->iter_);
    PBMap ea2s = reverse(es2a);
    std::vector<PBMap> laterMapList(laterList.size());
    std::vector<GenPBExpr::VarMap> laterExternalsList(laterList.size());
    std::vector<PBMap> ls2aList(laterList.size()), depAllList(laterList.size());
    PBMap extConstraint, spDepAllUnion;
    for (auto &&[i, later, laterMap, laterExternals] :
         views::zip(views::ints(0, ranges::unreachable), laterList,
                    laterMapList, laterExternalsList)) {
        laterMap =
            makeAccMap(presburger, *later, iterDim, accDim, laterRelax_,
                       "later" + std::to_string((uint64_t)later->stmt_->id()),
                       laterExternals, noNeedToBeVars);
    }
    projectOutPrivateAxis(presburger, earlier, laterList, laterMapList,
                          iterDim);
    for (auto &&[i, later, laterMap, laterExternals, ls2a, depAll] :
         views::zip(views::ints(0, ranges::unreachable), laterList,
                    laterMapList, laterExternalsList, ls2aList, depAllList)) {
        if (laterMap.empty()) {
            continue;
        }
        ls2a = makeSerialToAll(presburger, iterDim, later->iter_);
        PBMap la2s = reverse(ls2a);

        depAll = subtract(applyRange(laterMap, reverse(earlierMap)), allEQ);

        // Constraints on dependences. They decide when there are dependence,
        // but do not affect the existence of accesses. They should be applied
        // only to `depAll`.
        depAll = intersect(std::move(depAll), eraseVarDefConstraint);
        depAll = intersect(std::move(depAll), noDepsConstraint);
        depAll = coalesce(std::move(depAll));

        // Constraints on external variables limit the possibility of certain
        // relations on those variables. If these constraints are violated, it
        // not only means that there is no dependence, but also that the
        // accesses are completely impossible. Therefore, they should be
        // considered as constraints on the universal set, applied to not only
        // `depAll`, but also `earlierMap` and `laterMap` when checking killing
        // cases.
        auto extConstraintLocal = makeExternalVarConstraint(
            presburger, later, earlier, laterExternals, earlierExternals,
            iterDim);
        extConstraint = extConstraint.isValid()
                            ? intersect(std::move(extConstraint),
                                        std::move(extConstraintLocal))
                            : std::move(extConstraintLocal);
        extConstraint = coalesce(std::move(extConstraint));

        PBMap spDepAll = applyDomain(depAll, std::move(la2s));
        spDepAllUnion = spDepAllUnion.isValid()
                            ? uni(std::move(spDepAllUnion), std::move(spDepAll))
                            : std::move(spDepAll);
    }
    if (!spDepAllUnion.isValid()) {
        return;
    }
    spDepAllUnion =
        coalesce(intersect(std::move(spDepAllUnion), extConstraint));

    PBMap serialLexGT = lexGT(spaceSetAlloc(presburger, 0, iterDim));
    PBMap serialEQ = identity(spaceAlloc(presburger, 0, iterDim, iterDim));
    PBMap ssDepAll = applyRange(spDepAllUnion, std::move(ea2s));
    PBMap ssDep = intersect(ssDepAll, std::move(serialLexGT));
    PBMap ssSelf = intersect(ssDepAll, std::move(serialEQ));
    PBMap spDep =
        coalesce(intersect(applyRange(std::move(ssDep), es2a), spDepAllUnion));
    PBMap spSelf = intersect(applyRange(std::move(ssSelf), std::move(es2a)),
                             std::move(spDepAllUnion));
    PBMap spNearest =
        uni(reverse(lexmin(reverse(std::move(spDep)))), std::move(spSelf));
    spNearest = coalesce(std::move(spNearest));

    for (auto &&[later, ls2a, laterMap, depAll] :
         views::zip(laterList, ls2aList, laterMapList, depAllList)) {
        if (depAll.isValid()) {
            checkAgainstCond(
                presburger, later, earlier, depAll,
                intersect(applyDomain(spNearest, std::move(ls2a)), depAll),
                laterMap, earlierMap, extConstraint, iterDim);
        }
    }
}

void AnalyzeDeps::genTasks() {
    // Store / ReduceTo -> Load : RAW
    if (depType_ & DEP_RAW) {
        for (auto &&read : readsAsLater_) {
            checkDepLatestEarlier(read, writesAsEarlier_);
        }
    }

    // Load -> Store / ReduceTo : WAR
    if (depType_ & DEP_WAR) {
        for (auto &&read : readsAsEarlier_) {
            checkDepEarliestLater(writesAsLater_, read);
        }
    }

    // Store    -> Store    : WAW
    // ReduceTo -> Store    : WAW, WAR
    // Store    -> ReduceTo : WAW, RAW
    // ReduceTo -> ReduceTo : WAW, RAW, WAR
    if (depType_ & DEP_WAW) {
        // Every Store checks its immediate predecessor, so we
        // do not have to check its follower
        for (auto &&write : writesAsLater_) {
            checkDepLatestEarlier(write, writesAsEarlier_);
        }
    } else {
        if (depType_ & DEP_RAW) {
            for (auto &&write : writesAsLater_) {
                if (write->op_->nodeType() == ASTNodeType::ReduceTo) {
                    checkDepLatestEarlier(write, writesAsEarlier_);
                }
            }
        }
        if (depType_ & DEP_WAR) {
            for (auto &&write : writesAsEarlier_) {
                if (write->op_->nodeType() == ASTNodeType::ReduceTo) {
                    checkDepEarliestLater(writesAsLater_, write);
                }
            }
        }
    }
}

PBMap Dependence::extraCheck(PBMap dep,
                             const NodeIDOrParallelScope &nodeOrParallel,
                             const DepDirection &dir) const {
    PBMap require;
    if (nodeOrParallel.isNode_) {
        require = self_.makeConstraintOfSingleLoop(
            presburger_, nodeOrParallel.id_, dir, iterDim_);
    } else {
        require = self_.makeConstraintOfParallelScope(
            presburger_, nodeOrParallel.parallel_, dir, iterDim_, later_,
            earlier_);
    }
    dep = intersect(std::move(dep), std::move(require));
    return dep;
}

void FindDeps::operator()(const Stmt &op, const FindDepsCallback &found) {
    if (direction_.empty()) {
        return;
    }

    if (mode_ != FindDepsMode::Dep) {
        noProjectOutPrivateAxis_ = true;
    }

    FindAllNoDeps noDepsFinder;
    noDepsFinder(op);

    // Number the iteration space coordinates variable by variable, in order to
    // make the space more compact, so can be better coalesced
    auto defs = findAllStmt(
        op, [](const Stmt &s) { return s->nodeType() == ASTNodeType::VarDef; });
    std::vector<FindAccessPoint> finders;
    finders.reserve(defs.size());
    for (auto &&def : defs) {
        finders.emplace_back(def->id(), type_, accFilter_);
    }
    exceptSafeParallelFor<size_t>(
        0, finders.size(), 1, [&](size_t i) { finders[i].doFind(op); },
        omp_sched_dynamic);
    if (scope2CoordCallback_) {
        for (auto &&[def, accFinder] : views::zip(defs, finders)) {
            scope2CoordCallback_(def->id(), accFinder.scope2coord());
        }
    }

    auto variantExpr = LAZY(findLoopVariance(op).first);

    std::vector<std::function<void()>> tasks;
    std::vector<AnalyzeDeps> analyzers;
    analyzers.reserve(defs.size());
    for (auto &&accFinder : finders) {
        analyzers.emplace_back(
            accFinder.reads(), accFinder.writes(), accFinder.scope2coord(),
            noDepsFinder.results(), variantExpr, direction_, found, mode_,
            type_, earlierFilter_, laterFilter_, filter_, ignoreReductionWAW_,
            eraseOutsideVarDef_, noProjectOutPrivateAxis_);
        auto &analyzer = analyzers.back();
        analyzer.genTasks();
        for (auto &&task : analyzer.tasks()) {
            tasks.emplace_back(task);
        }
    }
    exceptSafeParallelFor<size_t>(
        0, tasks.size(), 1, [&](size_t i) { tasks[i](); }, omp_sched_dynamic);
}

bool FindDeps::exists(const Stmt &op) {
    struct DepExistsExcept {};
    try {
        (*this)(op, unsyncFunc([](const Dependence &dep) {
                    throw DepExistsExcept();
                }));
    } catch (const DepExistsExcept &e) {
        return true;
    }
    return false;
}

std::ostream &operator<<(std::ostream &_os, const Dependence &dep) {
    std::ostringstream os;
    os << "Dependence ";
    os << (dep.later()->nodeType() == ASTNodeType::Load ? "READ " : "WRITE ")
       << dep.later();
    if (dep.later()->isExpr()) {
        os << " in " << dep.later_.stmt_;
    }
    os << " after ";
    os << (dep.earlier()->nodeType() == ASTNodeType::Load ? "READ " : "WRITE ")
       << dep.earlier();
    if (dep.earlier()->isExpr()) {
        os << " in " << dep.earlier_.stmt_;
    }
    bool first = true;
    for (auto &&[scope, dir] : dep.dir_) {
        os << (first ? " along " : " and ");
        first = false;
        if (scope.isNode_) {
            os << scope.id_;
        } else {
            os << scope.parallel_;
        }
    }
    std::string str = os.str();
    std::erase(str, '\n');
    return _os << str;
}

} // namespace freetensor
